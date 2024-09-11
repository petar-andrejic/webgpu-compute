use std::{collections::VecDeque, sync::{atomic::{AtomicBool, Ordering}, Arc}};

use parking_lot::Mutex;

use super::ChannelError;

#[derive(Debug)]
struct Queue<T> {
    queue: Mutex<VecDeque<T>>,
    sender_closed: AtomicBool,
    receiver_closed: AtomicBool
}

impl<T> Queue<T> {
    fn new() -> Self {
        let queue = Mutex::new(VecDeque::new());
        Self {
            queue,
            sender_closed: false.into(),
            receiver_closed: false.into()
        }
    }
}

#[derive(Debug)]
pub struct Sender<T> {
    inner: Arc<Queue<T>>
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        self.inner.sender_closed.store(true, Ordering::Release)
    }
}

impl<T> Sender<T> {
    pub fn try_send(&self, value: T) -> Result<(), ChannelError> {
        if self.inner.receiver_closed.load(Ordering::Acquire) {
            return Err(ChannelError::ReceiverClosed);
        }

        self.inner.queue.lock().push_back(value);
        Ok(())
    }   

    pub fn send(&self, value: T) {
        self.try_send(value).unwrap();
    }
}

#[derive(Debug)]
pub struct Receiver<T> {
    inner: Arc<Queue<T>>
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        self.inner.receiver_closed.store(true, Ordering::Release)
    }
}

impl<T> Receiver<T> {
    pub fn try_receive(&self) -> Result<T, ChannelError> {
        let value = self.inner.queue.lock().pop_front();
        value.ok_or_else(||{if self.inner.sender_closed.load(Ordering::Acquire) {
            ChannelError::SenderClosed
        } else {
            ChannelError::Empty
        }})
    }

    pub fn receive(&self) -> T {
        self.try_receive().unwrap()
    }
}

pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let s_inner = Arc::new(Queue::new());
    let r_inner = s_inner.clone();
    (Sender {inner: s_inner}, Receiver {inner : r_inner})
}