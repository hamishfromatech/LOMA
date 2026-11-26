import type { Metadata, Viewport } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import { TooltipProvider } from "@/components/ui/tooltip"
import "./globals.css"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
})

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
})

export const metadata: Metadata = {
  title: "LOMA — Local Offline Mini Assistant",
  description:
    "A voice agent that runs entirely on your machine: faster-whisper, Needle 3, Qwen 3.5 and Phonon.",
}

export const viewport: Viewport = {
  themeColor: "#0b0b0a",
}

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`dark ${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full bg-background text-foreground">
        <a
          href="#loma-main"
          className="fixed left-3 top-3 z-[100] -translate-y-[4rem] rounded-full bg-loma-ink px-4 py-2 font-mono text-xs text-loma-bg transition-transform focus-visible:translate-y-0"
        >
          Skip to content
        </a>
        <TooltipProvider delay={200}>{children}</TooltipProvider>
      </body>
    </html>
  )
}
