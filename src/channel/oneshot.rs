use std::{
    cell::UnsafeCell,
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Poll, Waker},
};

use parking_lot::Mutex;

use super::ChannelError;

#[derive(Debug)]
struct Data<T> {
    value: UnsafeCell<Option<T>>,
    closed: AtomicBool,
    waker: Mutex<Option<Waker>>,
}

impl<T> Data<T> {
    fn new() -> Arc<Self> {
        let data = Self {
            value: UnsafeCell::new(None),
            closed: false.into(),
            waker: Mutex::new(None),
        };
        Arc::new(data)
    }
}

unsafe impl<T> Sync for Data<T> {}

#[derive(Debug)]
pub struct Sender<T> {
    data: Arc<Data<T>>,
}

#[derive(Debug)]
pub struct Receiver<T> {
    data: Arc<Data<T>>,
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        self.data.closed.store(true, Ordering::Release);
        self.data.waker.lock().as_ref().map(|w| w.wake_by_ref());
    }
}

impl<T> Sender<T> {
    pub fn send(self, val: T) {
        unsafe {
            let ptr = self.data.value.get();
            ptr.write(Some(val));
        }
        drop(self);
    }
}

impl<T> Receiver<T> {
    fn ready(&self) -> bool {
        self.data.closed.load(Ordering::Acquire)
    }
}

impl<T> Future for Receiver<T> {
    type Output = Result<T, ChannelError>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.ready() {
            let val = unsafe {
                let ptr = self.data.value.get();
                ptr.read()
            };
            Poll::Ready(val.ok_or(ChannelError::SenderClosed))
        } else {
            self.data.waker.lock().replace(cx.waker().clone());
            Poll::Pending
        }
    }
}

pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let s_data = Data::new();
    let r_data = s_data.clone();
    (Sender { data: s_data }, Receiver { data: r_data })
}
