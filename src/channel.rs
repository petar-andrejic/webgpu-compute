pub mod oneshot;
pub mod once_signal;
pub mod spsc;

#[derive(thiserror::Error, Debug)]
pub enum ChannelError {
    #[error("Tried to receive but sender was closed")]
    SenderClosed,
    #[error("Tried to send but receiver was closed")]
    ReceiverClosed,
    #[error("Channel is open but empty")]
    Empty
}