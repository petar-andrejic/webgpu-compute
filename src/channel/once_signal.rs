use std::{
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Poll, Waker},
};

use parking_lot::Mutex;

#[derive(Debug)]
struct Data {
    closed: AtomicBool,
    waker: Mutex<Option<Waker>>,
}

impl Data {
    fn new() -> Arc<Self> {
        let data = Self {
            closed: false.into(),
            waker: Mutex::new(None),
        };
        Arc::new(data)
    }
}

#[derive(Debug)]
pub struct Sender {
    data: Arc<Data>,
}

impl Drop for Sender {
    fn drop(&mut self) {
        self.data.closed.store(true, Ordering::Release);
        if let Some(waker) = self.data.waker.lock().as_ref() {
            waker.wake_by_ref()
        }
    }
}

impl Sender {
    pub fn send(self) {
        drop(self);
    }
}

#[derive(Debug)]
pub struct Receiver {
    data: Arc<Data>,
}

impl Receiver {
    fn ready(&self) -> bool {
        self.data.closed.load(Ordering::Acquire)
    }
}

impl Future for Receiver {
    type Output = ();

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.ready() {
            Poll::Ready(())
        } else {
            self.data.waker.lock().replace(cx.waker().clone());
            Poll::Pending
        }
    }
}

pub fn channel() -> (Sender, Receiver) {
    let s_data = Data::new();
    let r_data = s_data.clone();
    (Sender { data: s_data }, Receiver { data: r_data })
}
