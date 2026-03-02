import "./globals.css";
import FlowFieldBackground from "./components/flow-field-background";

export const metadata = {
  title: "ESP32-S3 Object Recognition Dashboard",
  description: "Futuristic monitoring dashboard for ESP32-S3 embedded ML object recognition system with MobileNetV2 INT8 inference",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <FlowFieldBackground
          color="#00d4ff"
          trailOpacity={0.12}
          particleCount={600}
          speed={0.8}
        />
        {children}
      </body>
    </html>
  );
}
