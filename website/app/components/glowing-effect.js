"use client";

import { memo, useCallback, useEffect, useRef } from "react";
import { animate } from "motion/react";

const GlowingEffect = memo(function GlowingEffect({
    blur = 0,
    inactiveZone = 0.7,
    proximity = 0,
    spread = 20,
    variant = "default",
    glow = false,
    className = "",
    movementDuration = 2,
    borderWidth = 1,
    disabled = true,
}) {
    const containerRef = useRef(null);
    const lastPosition = useRef({ x: 0, y: 0 });
    const animationFrameRef = useRef(0);

    const handleMove = useCallback(
        (e) => {
            if (!containerRef.current) return;

            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }

            animationFrameRef.current = requestAnimationFrame(() => {
                const element = containerRef.current;
                if (!element) return;

                const { left, top, width, height } = element.getBoundingClientRect();
                const mouseX = e?.x ?? lastPosition.current.x;
                const mouseY = e?.y ?? lastPosition.current.y;

                if (e) {
                    lastPosition.current = { x: mouseX, y: mouseY };
                }

                const center = [left + width * 0.5, top + height * 0.5];
                const distanceFromCenter = Math.hypot(
                    mouseX - center[0],
                    mouseY - center[1]
                );
                const inactiveRadius = 0.5 * Math.min(width, height) * inactiveZone;

                if (distanceFromCenter < inactiveRadius) {
                    element.style.setProperty("--active", "0");
                    return;
                }

                const isActive =
                    mouseX > left - proximity &&
                    mouseX < left + width + proximity &&
                    mouseY > top - proximity &&
                    mouseY < top + height + proximity;

                element.style.setProperty("--active", isActive ? "1" : "0");

                if (!isActive) return;

                const currentAngle =
                    parseFloat(element.style.getPropertyValue("--start")) || 0;
                let targetAngle =
                    (180 * Math.atan2(mouseY - center[1], mouseX - center[0])) /
                    Math.PI +
                    90;

                const angleDiff = ((targetAngle - currentAngle + 180) % 360) - 180;
                const newAngle = currentAngle + angleDiff;

                animate(currentAngle, newAngle, {
                    duration: movementDuration,
                    ease: [0.16, 1, 0.3, 1],
                    onUpdate: (value) => {
                        element.style.setProperty("--start", String(value));
                    },
                });
            });
        },
        [inactiveZone, proximity, movementDuration]
    );

    useEffect(() => {
        if (disabled) return;

        const handleScroll = () => handleMove();
        const handlePointerMove = (e) => handleMove(e);

        window.addEventListener("scroll", handleScroll, { passive: true });
        document.body.addEventListener("pointermove", handlePointerMove, {
            passive: true,
        });

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
            window.removeEventListener("scroll", handleScroll);
            document.body.removeEventListener("pointermove", handlePointerMove);
        };
    }, [handleMove, disabled]);

    const gradient =
        variant === "white"
            ? `repeating-conic-gradient(
          from 236.84deg at 50% 50%,
          #000,
          #000 calc(25% / var(--repeating-conic-gradient-times))
        )`
            : `radial-gradient(circle, #00d4ff 10%, #00d4ff00 20%),
         radial-gradient(circle at 40% 40%, #a855f7 5%, #a855f700 15%),
         radial-gradient(circle at 60% 60%, #10b981 10%, #10b98100 20%),
         radial-gradient(circle at 40% 60%, #3b82f6 10%, #3b82f600 20%),
         repeating-conic-gradient(
           from 236.84deg at 50% 50%,
           #00d4ff 0%,
           #a855f7 calc(25% / var(--repeating-conic-gradient-times)),
           #10b981 calc(50% / var(--repeating-conic-gradient-times)),
           #3b82f6 calc(75% / var(--repeating-conic-gradient-times)),
           #00d4ff calc(100% / var(--repeating-conic-gradient-times))
         )`;

    return (
        <>
            <div
                className={`glow-border-static ${glow ? "glow-visible" : ""} ${disabled ? "glow-show" : ""} ${variant === "white" ? "glow-white" : ""}`}
            />
            <div
                ref={containerRef}
                style={{
                    "--blur": `${blur}px`,
                    "--spread": spread,
                    "--start": "0",
                    "--active": "0",
                    "--glowingeffect-border-width": `${borderWidth}px`,
                    "--repeating-conic-gradient-times": "5",
                    "--gradient": gradient,
                }}
                className={`glow-container ${glow ? "glow-visible" : ""} ${blur > 0 ? "glow-blur" : ""} ${disabled ? "glow-hidden" : ""} ${className}`}
            >
                <div className="glow-inner" />
            </div>
        </>
    );
});

export { GlowingEffect };
