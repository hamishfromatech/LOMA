import { cn } from "@/lib/utils"

/**
 * Pulsing placeholder block used in loading states. Kept deliberately
 * minimal — a shimmer sweep is noise on a slow device.
 */
export function Skeleton({
  className,
  role = "presentation",
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      role={role}
      aria-hidden="true"
      className={cn(
        "rounded-lg bg-loma-raise/60 animate-pulse motion-reduce:animate-none",
        className,
      )}
      {...props}
    />
  )
}

/** A stack of skeleton rows — a simple "the data is loading" rhythm. */
export function SkeletonRows({ count = 3, className }: { count?: number; className?: string }) {
  return (
    <div aria-hidden="true" className={cn("space-y-2", className)}>
      {Array.from({ length: count }, (_, i) => (
        <Skeleton
          key={i}
          className={i === count - 1 ? "w-2/3" : "w-full"}
          // h-3 is a good stand-in for a line of text
          style={{ height: 12 }}
        />
      ))}
    </div>
  )
}
