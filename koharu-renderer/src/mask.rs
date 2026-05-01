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
        Self {
            width,
            height,
            data,
        }
    }

    pub fn contains_rect(&self, x: f32, y: f32, w: f32, h: f32) -> bool {
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let x1 = (x + w).ceil() as isize;
        let y1 = (y + h).ceil() as isize;

        // Check 4 corners just like MangaTranslator to allow slight boundary overlap
        let points = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];

        for (px, py) in points {
            let px = px.clamp(0, (self.width as isize) - 1) as usize;
            let py = py.clamp(0, (self.height as isize) - 1) as usize;

            if !self.data[py * self.width + px] {
                return false;
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
                    line.baseline.0 - layout.line_height * 0.5,
                    line.baseline.1,
                    layout.line_height,
                    line.advance,
                )
            } else {
                (
                    line.baseline.0,
                    line.baseline.1 - layout.ascent,
                    line.advance,
                    layout.line_height,
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
