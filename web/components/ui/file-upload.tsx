"use client"

import { cn } from "@/lib/utils"
import {
  Children,
  cloneElement,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react"
import { createPortal } from "react-dom"

type FileUploadContextValue = {
  isDragging: boolean
  inputRef: React.RefObject<HTMLInputElement | null>
  multiple?: boolean
  disabled?: boolean
}

const FileUploadContext = createContext<FileUploadContextValue | null>(null)

export type FileUploadProps = {
  onFilesAdded: (files: File[]) => void
  children: React.ReactNode
  multiple?: boolean
  accept?: string
  disabled?: boolean
}

function FileUpload({
  onFilesAdded,
  children,
  multiple = true,
  accept,
  disabled = false,
}: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const dragCounter = useRef(0)

  const matchesAccept = useCallback(
    (file: File) => {
      if (!accept) return true
      // Comma-separated mime types/patterns, e.g. "image/png,image/jpeg,image/webp"
      const patterns = accept.split(",").map((part) => part.trim().toLowerCase()).filter(Boolean)
      const type = file.type.toLowerCase()
      return patterns.some((pattern) => {
        if (pattern.endsWith("/*")) {
          const prefix = pattern.slice(0, -2)
          return type.startsWith(prefix)
        }
        return pattern === type
      })
    },
    [accept],
  )

  const matchesItem = useCallback(
    (item: DataTransferItem) => {
      if (!accept) return item.kind === "file"
      if (item.kind !== "file") return false
      const type = (item.type || "").toLowerCase()
      const patterns = accept.split(",").map((part) => part.trim().toLowerCase()).filter(Boolean)
      return patterns.some((pattern) => {
        if (pattern.endsWith("/*")) {
          return type.startsWith(pattern.slice(0, -2))
        }
        return pattern === type
      })
    },
    [accept],
  )

  const handleFiles = useCallback(
    (files: FileList) => {
      const matching = Array.from(files).filter(matchesAccept)
      if (matching.length === 0) return
      if (multiple) {
        onFilesAdded(matching)
      } else {
        onFilesAdded(matching.slice(0, 1))
      }
    },
    [multiple, matchesAccept, onFilesAdded],
  )

  useEffect(() => {
    const handleDrag = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
    }

    const handleDragIn = (e: DragEvent) => {
      handleDrag(e)
      // Sniff the payload — accept only when a matching file is present, so
      // a PDF drop doesn't flash "Drop an image".
      const items = Array.from(e.dataTransfer?.items ?? [])
      const matching = items.some((item) => item.kind === "file" && matchesItem(item))
      if (matching) {
        dragCounter.current++
        setIsDragging(true)
      }
    }

    const handleDragOut = (e: DragEvent) => {
      handleDrag(e)
      dragCounter.current--
      if (dragCounter.current === 0) setIsDragging(false)
    }

    const handleDrop = (e: DragEvent) => {
      handleDrag(e)
      setIsDragging(false)
      dragCounter.current = 0
      if (e.dataTransfer?.files.length) {
        handleFiles(e.dataTransfer.files)
      }
    }

    window.addEventListener("dragenter", handleDragIn)
    window.addEventListener("dragleave", handleDragOut)
    window.addEventListener("dragover", handleDrag)
    window.addEventListener("drop", handleDrop)

    return () => {
      window.removeEventListener("dragenter", handleDragIn)
      window.removeEventListener("dragleave", handleDragOut)
      window.removeEventListener("dragover", handleDrag)
      window.removeEventListener("drop", handleDrop)
    }
  }, [handleFiles, matchesItem])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      handleFiles(e.target.files)
      e.target.value = ""
    }
  }

  return (
    <FileUploadContext.Provider
      value={{ isDragging, inputRef, multiple, disabled }}
    >
      <input
        type="file"
        ref={inputRef}
        onChange={handleFileSelect}
        className="hidden"
        multiple={multiple}
        accept={accept}
        aria-hidden
        disabled={disabled}
      />
      {children}
    </FileUploadContext.Provider>
  )
}

export type FileUploadTriggerProps =
  React.ComponentPropsWithoutRef<"button"> & {
    asChild?: boolean
  }

function FileUploadTrigger({
  asChild = false,
  className,
  children,
  ...props
}: FileUploadTriggerProps) {
  const context = useContext(FileUploadContext)
  const handleClick = () => context?.inputRef.current?.click()

  if (asChild) {
    const child = Children.only(children) as React.ReactElement<
      React.HTMLAttributes<HTMLElement>
    >
    return cloneElement(child, {
      ...props,
      role: "button",
      className: cn(className, child.props.className),
      onClick: (e: React.MouseEvent) => {
        e.stopPropagation()
        handleClick()
        child.props.onClick?.(e as React.MouseEvent<HTMLElement>)
      },
    })
  }

  return (
    <button
      type="button"
      className={className}
      onClick={handleClick}
      {...props}
    >
      {children}
    </button>
  )
}

type FileUploadContentProps = React.HTMLAttributes<HTMLDivElement>

function FileUploadContent({
  className,
  children,
  ...props
}: FileUploadContentProps) {
  const context = useContext(FileUploadContext)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    return () => setMounted(false)
  }, [])

  if (!context?.isDragging || !mounted || context?.disabled) {
    return null
  }

  const content = (
    <div
      className={cn(
        "bg-background/80 fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm pointer-events-none",
        "animate-in fade-in-0 slide-in-from-bottom-10 zoom-in-90 duration-150",
        className
      )}
      {...props}
    >
      {children}
    </div>
  )

  return createPortal(content, document.body)
}

export { FileUpload, FileUploadTrigger, FileUploadContent }
