"use client";

import { useState, useRef, useEffect, useCallback, memo } from "react";

/**
 * SnappySlider — A precision slider with snap-to markers, direct numeric input,
 * drag/touch support, out-of-bounds indication, and double-click reset.
 * Adapted for vanilla CSS (no Tailwind / shadcn).
 */
const SnappySlider = memo(function SnappySlider({
    values = [],
    defaultValue = 0.5,
    value,
    snapping = true,
    min: providedMin,
    max: providedMax,
    step = 0.01,
    onChange,
    label = "Value",
    prefix = "",
    suffix = "",
    snappingThreshold = 0.02,
}) {
    const sliderRef = useRef(null);

    const allVals = [...new Set([...values, defaultValue])].sort((a, b) => a - b);
    const inputMin = providedMin ?? Math.min(...allVals);
    const inputMax = providedMax ?? Math.max(...allVals);

    const sliderValues =
        providedMin !== undefined && providedMax !== undefined
            ? allVals.filter((v) => v >= providedMin && v <= providedMax)
            : allVals;
    const sliderMin = Math.min(...sliderValues);
    const sliderMax = Math.max(...sliderValues);

    const fmt = (v) => {
        const dec = step.toString().split(".")[1]?.length || 0;
        return Number(v).toFixed(dec);
    };

    const [internalValue, setInternalValue] = useState(defaultValue);
    const current = value ?? internalValue;
    const [inputValue, setInputValue] = useState(fmt(current));

    const isOob = current < sliderMin || current > sliderMax;
    const pct =
        ((Math.min(Math.max(current, sliderMin), sliderMax) - sliderMin) /
            (sliderMax - sliderMin)) *
        100;

    useEffect(() => {
        if (value !== undefined) {
            setInternalValue(value);
            setInputValue(fmt(value));
        }
    }, [value]);

    const fire = (v) => {
        setInternalValue(v);
        setInputValue(fmt(v));
        onChange?.(v);
    };

    const handleInputChange = (e) => setInputValue(e.target.value);

    const handleInputBlur = () => {
        const n = Number(inputValue);
        if (isNaN(n)) {
            setInputValue(fmt(current));
        } else {
            const c = Math.max(inputMin, Math.min(inputMax, n));
            const s = Math.round(c / step) * step;
            fire(s);
        }
    };

    const interact = useCallback(
        (clientX) => {
            const el = sliderRef.current;
            if (!el) return;
            const rect = el.getBoundingClientRect();
            const prc = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
            const raw = prc * (sliderMax - sliderMin) + sliderMin;

            if (snapping) {
                const snap = [...new Set([...allVals, current])].sort((a, b) => a - b);
                const closest = snap.reduce((prev, curr) =>
                    Math.abs(curr - raw) < Math.abs(prev - raw) ? curr : prev
                );
                if (Math.abs(closest - raw) <= snappingThreshold) {
                    fire(closest);
                    return;
                }
            }
            const s = Math.round(raw / step) * step;
            fire(Math.max(sliderMin, Math.min(sliderMax, s)));
        },
        [sliderMin, sliderMax, allVals, current, step, snapping, snappingThreshold]
    );

    useEffect(() => {
        const el = sliderRef.current;
        if (!el) return;

        const onMouseDown = (e) => {
            e.preventDefault();
            interact(e.clientX);
            document.body.style.userSelect = "none";

            const onMove = (e) => interact(e.clientX);
            const onUp = () => {
                document.removeEventListener("mousemove", onMove);
                document.body.style.userSelect = "";
            };
            document.addEventListener("mousemove", onMove);
            document.addEventListener("mouseup", onUp, { once: true });
        };

        const onTouchStart = (e) => {
            e.preventDefault();
            interact(e.touches[0].clientX);
            const onMove = (e) => interact(e.touches[0].clientX);
            document.addEventListener("touchmove", onMove, { passive: false });
            document.addEventListener("touchend", () => document.removeEventListener("touchmove", onMove), { once: true });
        };

        el.addEventListener("mousedown", onMouseDown);
        el.addEventListener("touchstart", onTouchStart, { passive: false });
        return () => {
            el.removeEventListener("mousedown", onMouseDown);
            el.removeEventListener("touchstart", onTouchStart);
            document.body.style.userSelect = "";
        };
    }, [interact]);

    useEffect(() => {
        const el = sliderRef.current;
        if (!el) return;
        const dbl = () => fire(defaultValue);
        el.addEventListener("dblclick", dbl);
        return () => el.removeEventListener("dblclick", dbl);
    }, [defaultValue]);

    const handleKeyDown = (e) => {
        if (e.key === "ArrowUp" || e.key === "ArrowDown") {
            e.preventDefault();
            const v = Number(inputValue);
            if (isNaN(v)) return;
            const nv = v + (e.key === "ArrowUp" ? step : -step);
            const cl = Math.max(sliderMin, Math.min(sliderMax, nv));
            fire(cl);
        }
        if (e.key === "Enter") e.target.blur();
    };

    return (
        <div className="snappy-slider">
            {/* Header */}
            <div className="snappy-header">
                <label className="snappy-label">{label}</label>
                <div className="snappy-input-wrap">
                    {prefix && <span className="snappy-affix">{prefix}</span>}
                    <input
                        type="number"
                        className={`snappy-input ${isOob ? "snappy-oob" : ""}`}
                        value={inputValue}
                        onChange={handleInputChange}
                        onBlur={handleInputBlur}
                        onKeyDown={handleKeyDown}
                        step={step}
                    />
                    {suffix && <span className="snappy-affix">{suffix}</span>}
                </div>
            </div>

            {/* Track */}
            <div className="snappy-track-area" ref={sliderRef}>
                <div className="snappy-track">
                    {/* Fill */}
                    <div className="snappy-fill" style={{ width: `${pct}%` }} />

                    {/* Markers */}
                    {sliderValues.map((mark, i) => {
                        const mp = ((mark - sliderMin) / (sliderMax - sliderMin)) * 100;
                        if (mp < 0 || mp > 100) return null;
                        return (
                            <div
                                key={`${mark}-${i}`}
                                className="snappy-marker"
                                style={{ left: `${mp}%` }}
                            />
                        );
                    })}
                </div>

                {/* Thumb */}
                <div
                    className={`snappy-thumb ${isOob ? "snappy-oob" : ""}`}
                    style={{ left: `${pct}%` }}
                >
                    <div className="snappy-thumb-arrow" />
                    <div className="snappy-thumb-square" />
                    <div className="snappy-thumb-label">
                        {isOob
                            ? current < sliderMin
                                ? `<${fmt(sliderMin)}`
                                : `>${fmt(sliderMax)}`
                            : fmt(current)}
                    </div>
                </div>
            </div>
        </div>
    );
});

export { SnappySlider };
