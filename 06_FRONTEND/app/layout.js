import "./globals.css";

export const metadata = {
  title: "NEREUS",
  description:
    "Intelligent freight forecasting and vessel chartering decision support.",
};

export default function RootLayout({
  children,
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}