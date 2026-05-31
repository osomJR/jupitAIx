"use client";

import { useCallback, useMemo, useRef } from "react";
import {
  LiveKitRoom,
  RoomAudioRenderer,
  VideoConference,
} from "@livekit/components-react";

/**
 * Reusable LiveKit call room wrapper.
 *
 * Props:
 * - serverUrl: LiveKit WebSocket URL, e.g. "wss://your-project.livekit.cloud"
 * - token: backend-generated LiveKit participant token
 * - roomName: backend-generated call room name
 * - onLeave: async/sync callback fired after LiveKit disconnects
 *
 * Important:
 * Import LiveKit styles globally once, preferably in app/layout.js:
 *
 *   import "@livekit/components-styles";
 */
export default function TeamCallRoom({
  serverUrl,
  token,
  roomName,
  onLeave,
}) {
  const leavingRef = useRef(false);

  const canConnect = Boolean(serverUrl && token);

  const displayRoomName = useMemo(() => {
    if (!roomName) return "Team call";

    return String(roomName)
      .replace(/^org-/, "Organization ")
      .replaceAll("-", " ");
  }, [roomName]);

  const handleDisconnected = useCallback(async () => {
    if (leavingRef.current) {
      return;
    }

    leavingRef.current = true;

    try {
      await onLeave?.();
    } finally {
      leavingRef.current = false;
    }
  }, [onLeave]);

  if (!canConnect) {
    return (
      <section className="rounded-3xl border border-red-400/30 bg-red-400/10 p-6 text-red-100">
        <h2 className="text-lg font-semibold">Call unavailable</h2>
        <p className="mt-2 text-sm text-red-100/80">
          Missing LiveKit server URL or participant token. Start or join the call
          again to request a fresh token.
        </p>
      </section>
    );
  }

  return (
    <section className="overflow-hidden rounded-3xl border app-surface-strong">
      <div className="flex flex-col gap-2 border-b border-[var(--app-border)] px-5 py-4 md:flex-row md:items-center md:justify-between">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] app-text-soft">
            Live call
          </p>
          <h2 className="truncate text-lg font-semibold app-text">
            {displayRoomName}
          </h2>
        </div>

        <div className="rounded-full border border-[var(--app-border)] px-3 py-1 text-xs font-semibold app-text-soft">
          Connected by LiveKit
        </div>
      </div>

      <div className="h-[70vh] min-h-[32rem] bg-black">
        <LiveKitRoom
          serverUrl={serverUrl}
          token={token}
          connect
          audio
          video
          onDisconnected={handleDisconnected}
          data-lk-theme="default"
          className="h-full"
        >
          <VideoConference />
          <RoomAudioRenderer />
        </LiveKitRoom>
      </div>
    </section>
  );
}
