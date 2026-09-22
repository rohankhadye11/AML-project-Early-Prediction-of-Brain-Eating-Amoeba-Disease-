import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PAM Early Prediction Portal | Clinical CDS",
  description: "Multimodal Clinical Decision Support System for Primary Amoebic Meningoencephalitis",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
