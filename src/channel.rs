pub mod oneshot;
pub mod once_signal;

#[derive(thiserror::Error, Debug)]
pub enum ChannelError {
    #[error("Channel closed before send")]
    SenderClosed
}