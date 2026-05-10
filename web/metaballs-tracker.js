/**
 * Custom metaball tracker that simulates movement and exposes positions
 * for placing images inside metaballs.
 */

class MetaballTracker {
    constructor(options = {}) {
        this.numMetaballs = options.numMetaballs || 10;
        this.minRadius = options.minRadius || 5;
        this.maxRadius = options.maxRadius || 15;
        this.speed = options.speed || 5.0;
        this.canvasWidth = options.canvasWidth || window.innerWidth;
        this.canvasHeight = options.canvasHeight || window.innerHeight;
        
        this.metaballs = [];
        this.initMetaballs();
    }

    initMetaballs() {
        for (let i = 0; i < this.numMetaballs; i++) {
            this.metaballs.push({
                x: Math.random() * 100,
                y: Math.random() * 100,
                vx: (Math.random() - 0.5) * this.speed,
                vy: (Math.random() - 0.5) * this.speed,
                r: this.minRadius + Math.random() * (this.maxRadius - this.minRadius)
            });
        }
    }

    update() {
        this.metaballs.forEach(mb => {
            // Update position
            mb.x += mb.vx * 0.01;
            mb.y += mb.vy * 0.01;

            // Bounce off edges
            if (mb.x < 0 || mb.x > 100) {
                mb.vx *= -1;
                mb.x = Math.max(0, Math.min(100, mb.x));
            }
            if (mb.y < 0 || mb.y > 100) {
                mb.vy *= -1;
                mb.y = Math.max(0, Math.min(100, mb.y));
            }
        });
    }

    getPositions() {
        return this.metaballs.map(mb => ({
            x: mb.x,
            y: mb.y,
            radius: mb.r
        }));
    }

    setCanvasSize(width, height) {
        this.canvasWidth = width;
        this.canvasHeight = height;
    }
}
