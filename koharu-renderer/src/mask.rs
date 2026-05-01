use crate::layout::LayoutRun;

/// Represents a 2D collision mask where pixels inside the balloon are `true` (valid),
/// and pixels outside are `false` (invalid).
#[derive(Debug, Clone)]
pub struct CollisionMask {
    pub width: usize,
    pub height: usize,
    /// Row-major order grid of booleans.
    pub data: Vec<bool>,
}

impl CollisionMask {
    pub fn new(width: usize, height: usize, data: Vec<bool>) -> Self {
        assert_eq!(width * height, data.len());
        Self { width, height, data }
    }

    /// Tests if a given rectangle is completely inside the mask's "true" (valid) region.
    pub fn contains_rect(&self, x: f32, y: f32, w: f32, h: f32) -> bool {
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let x1 = (x + w).ceil() as isize;
        let y1 = (y + h).ceil() as isize;

        if x0 < 0 || y0 < 0 || x1 > self.width as isize || y1 > self.height as isize {
            return false; // Out of bounds
        }

        for py in y0..y1 {
            let row_offset = (py as usize) * self.width;
            for px in x0..x1 {
                if !self.data[row_offset + px as usize] {
                    return false;
                }
            }
        }
        true
    }

    /// Tests if a LayoutRun collides with the mask (i.e. falls outside the valid 'true' region).
    /// `offset_x` and `offset_y` represent the top-left coordinate where the LayoutRun is placed.
    pub fn collides_with(&self, layout: &LayoutRun, offset_x: f32, offset_y: f32) -> bool {
        for line in &layout.lines {
            // Conservative bounding box estimation for the line
            let is_vertical = line.direction == harfrust::Direction::TopToBottom;
            
            let (bx, by, bw, bh) = if is_vertical {
                (
                    line.baseline.0 - layout.font_size * 0.5,
                    line.baseline.1,
                    layout.font_size,
                    line.advance
                )
            } else {
                (
                    line.baseline.0,
                    line.baseline.1 - layout.font_size,
                    line.advance,
                    layout.font_size * 1.2
                )
            };

            let abs_x = offset_x + bx;
            let abs_y = offset_y + by;

            if !self.contains_rect(abs_x, abs_y, bw, bh) {
                return true; // Collision found!
            }
        }
        false
    }
}
