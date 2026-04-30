use std::fmt;

#[derive(Debug, Clone)]
pub enum Error {
    NumbersOfElementsError,
    PositionValueError,
    ShapeMismatchError,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::NumbersOfElementsError => write!(f, "Error: elements mismatch shape"),
            Error::PositionValueError => write!(f, "Error: index out of bounds"),
            Error::ShapeMismatchError => write!(f, "Error: shapes don't match"),
        }
    }
}
