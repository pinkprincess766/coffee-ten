use std::fmt;

/// Static tensor shape, row-major. An empty shape is a scalar.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: Vec<usize>) -> Shape {
        Shape(dims)
    }

    pub fn scalar() -> Shape {
        Shape(Vec::new())
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// NumPy broadcasting of two shapes.
    pub fn broadcast(&self, other: &Shape) -> Result<Shape, BroadcastError> {
        let rank = self.rank().max(other.rank());
        let mut dims = Vec::with_capacity(rank);
        for axis_from_end in 1..=rank {
            let left_dim = dim_from_end(&self.0, axis_from_end);
            let right_dim = dim_from_end(&other.0, axis_from_end);
            let dim = if left_dim == right_dim {
                left_dim
            } else if left_dim == 1 {
                right_dim
            } else if right_dim == 1 {
                left_dim
            } else {
                return Err(BroadcastError {
                    left: self.clone(),
                    right: other.clone(),
                    axis_from_end,
                });
            };
            dims.push(dim);
        }
        dims.reverse();
        Ok(Shape(dims))
    }
}

/// Size of the axis `axis_from_end` counted from the end (1 is the last axis).
/// Axes missing on the left are size 1.
fn dim_from_end(dims: &[usize], axis_from_end: usize) -> usize {
    if axis_from_end <= dims.len() {
        dims[dims.len() - axis_from_end]
    } else {
        1
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.as_slice() {
            [] => write!(f, "()"),
            [dim] => write!(f, "({dim},)"),
            dims => {
                let text: Vec<String> = dims.iter().map(usize::to_string).collect();
                write!(f, "({})", text.join(", "))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BroadcastError {
    pub left: Shape,
    pub right: Shape,
    /// Axis counted from the end: 1 is the last axis.
    pub axis_from_end: usize,
}

impl fmt::Display for BroadcastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let left_dim = dim_from_end(&self.left.0, self.axis_from_end);
        let right_dim = dim_from_end(&self.right.0, self.axis_from_end);
        write!(
            f,
            "cannot broadcast shapes {} and {}: {left_dim} vs {right_dim} at axis -{}",
            self.left, self.right, self.axis_from_end
        )
    }
}

impl std::error::Error for BroadcastError {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn shape(dims: &[usize]) -> Shape {
        Shape::new(dims.to_vec())
    }

    #[test]
    fn numpy_docs_examples_broadcast() {
        assert_eq!(
            shape(&[8, 1, 6, 1]).broadcast(&shape(&[7, 1, 5])),
            Ok(shape(&[8, 7, 6, 5]))
        );
        assert_eq!(shape(&[5, 4]).broadcast(&shape(&[1])), Ok(shape(&[5, 4])));
        assert_eq!(shape(&[5, 4]).broadcast(&shape(&[4])), Ok(shape(&[5, 4])));
        assert_eq!(
            shape(&[15, 3, 5]).broadcast(&shape(&[15, 1, 5])),
            Ok(shape(&[15, 3, 5]))
        );
        assert_eq!(
            shape(&[15, 3, 5]).broadcast(&shape(&[3, 5])),
            Ok(shape(&[15, 3, 5]))
        );
        assert_eq!(
            shape(&[15, 3, 5]).broadcast(&shape(&[3, 1])),
            Ok(shape(&[15, 3, 5]))
        );
    }

    #[test]
    fn scalar_broadcasts_to_anything() {
        assert_eq!(
            Shape::scalar().broadcast(&shape(&[3, 4])),
            Ok(shape(&[3, 4]))
        );
        assert_eq!(
            Shape::scalar().broadcast(&Shape::scalar()),
            Ok(Shape::scalar())
        );
    }

    #[test]
    fn size_zero_is_not_max_of_zero_and_one() {
        assert_eq!(shape(&[0]).broadcast(&shape(&[1])), Ok(shape(&[0])));
        assert_eq!(
            shape(&[0])
                .broadcast(&shape(&[3]))
                .unwrap_err()
                .axis_from_end,
            1
        );
    }

    #[test]
    fn mismatch_reports_axis_from_end() {
        let err = shape(&[3]).broadcast(&shape(&[4])).unwrap_err();
        assert_eq!(err.axis_from_end, 1);
        assert_eq!(
            err.to_string(),
            "cannot broadcast shapes (3,) and (4,): 3 vs 4 at axis -1"
        );
    }

    #[test]
    fn first_mismatch_is_scanned_from_the_end() {
        // Axis -1 is 1 vs 3 (fine); axis -2 is 2 vs 4 (the first mismatch).
        let err = shape(&[2, 1]).broadcast(&shape(&[8, 4, 3])).unwrap_err();
        assert_eq!(err.axis_from_end, 2);
        assert_eq!(err.left, shape(&[2, 1]));
        assert_eq!(err.right, shape(&[8, 4, 3]));
        // Both axes mismatch: the one closer to the end is reported.
        let err = shape(&[2, 3]).broadcast(&shape(&[4, 5])).unwrap_err();
        assert_eq!(err.axis_from_end, 1);
    }

    #[test]
    fn shape_display_uses_numpy_tuple_style() {
        assert_eq!(Shape::scalar().to_string(), "()");
        assert_eq!(shape(&[3]).to_string(), "(3,)");
        assert_eq!(shape(&[8, 7, 6, 5]).to_string(), "(8, 7, 6, 5)");
    }

    fn shape_strategy() -> impl Strategy<Value = Shape> {
        prop::collection::vec(0usize..=4, 0..=5).prop_map(Shape::new)
    }

    /// `a` and a `b` that broadcasts to `a`: drop some leading dims of `a`,
    /// then replace some of the remaining dims with 1.
    fn shape_and_compatible_shape() -> impl Strategy<Value = (Shape, Shape)> {
        shape_strategy().prop_flat_map(|a| {
            let rank = a.rank();
            (
                Just(a),
                0..=rank,
                prop::collection::vec(any::<bool>(), rank),
            )
                .prop_map(|(a, dropped, replace_with_one)| {
                    let b_dims: Vec<usize> = a.dims()[dropped..]
                        .iter()
                        .zip(replace_with_one)
                        .map(|(&dim, one)| if one { 1 } else { dim })
                        .collect();
                    (a, Shape::new(b_dims))
                })
        })
    }

    proptest! {
        #[test]
        fn broadcast_is_symmetric(a in shape_strategy(), b in shape_strategy()) {
            prop_assert_eq!(a.broadcast(&b).ok(), b.broadcast(&a).ok());
        }

        #[test]
        fn shape_broadcasts_with_itself(a in shape_strategy()) {
            prop_assert_eq!(a.broadcast(&a), Ok(a.clone()));
        }

        #[test]
        fn shape_broadcasts_with_scalar(a in shape_strategy()) {
            prop_assert_eq!(a.broadcast(&Shape::scalar()), Ok(a.clone()));
        }

        #[test]
        fn compatible_shape_broadcasts_to_the_larger(
            (a, b) in shape_and_compatible_shape()
        ) {
            prop_assert_eq!(a.broadcast(&b), Ok(a.clone()));
        }
    }
}
