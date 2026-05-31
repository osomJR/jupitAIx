"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  MessageCircle,
  RefreshCw,
  Send,
  UsersRound,
  Video,
} from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";

import TeamCallRoom from "@/components/team_call_room";
import { useAccount } from "@/components/account_provider";
import { useLanguage } from "@/components/language_provider";
import { useTeamRealtime } from "@/components/team_realtime_provider";
import {
  createConversation,
  getAccessToken,
  getConversationMessages,
  getOrganizationConversations,
  getOrganizationPresence,
  joinCall,
  leaveCall,
  startConversationCall,
} from "@/lib/api_client";

const copy = {
  en: {
    title: "Messages & Calls",
    subtitle:
      "Message and Call team members directly or on group chat",
    backToTeam: "Back to team",
    refresh: "Refresh",
    teamMembers: "Team members",
    message: "Message",
    call: "Call",
    videoCall: "Video call",
    you: "You",
    recentlyJoined: "Recently joined",
    groupWorkspace: "Team group chat",
    createGroupChat: "Create group chat",
    openGroupChat: "Open group chat",
    callGroup: "Video call group",
    ownerOnlyGroup:
      "Only the organization owner can create the team group chat.",
    noMembers: "No other active members found yet.",
    messages: "Messages",
    chooseConversation:
      "Choose a member or the team group chat to start messaging.",
    messagePlaceholder: "Write a message...",
    send: "Send",
    sending: "Sending...",
    realtimeConnecting: "Connecting...",
    messagePending: "Sending...",
    messageFailed: "Failed to send",
    startCall: "Start video call",
    joinCall: "Join video call",
    joining: "Joining...",
    starting: "Starting...",
    online: "Online",
    offline: "Offline",
    in_call: "In call",
    unavailableTitle: "Messages and calls are unavailable",
    unavailableDescription:
      "This workspace is only available to active Business or Enterprise organization members.",
    loading: "Loading messages...",
    directMessage: "Direct message",
    groupChat: "Group chat",
    memberChat: "Member chat",
    noConversations: "No conversations yet.",
    noMessagesOrCalls: "No messages or call logs yet.",
    creating: "Creating...",
    opening: "Opening...",
    noGroupYet: "No group chat yet.",
  },
  fr: {
    title: "Messages & appels",
    subtitle:
      "Envoyez des messages et appelez les membres directement ou dans le groupe.",
    backToTeam: "Retour à l’équipe",
    refresh: "Actualiser",
    teamMembers: "Membres de l’équipe",
    message: "Message",
    call: "Appel",
    videoCall: "Appel vidéo",
    you: "Vous",
    recentlyJoined: "Récemment rejoint",
    groupWorkspace: "Groupe de l’équipe",
    createGroupChat: "Créer le groupe",
    openGroupChat: "Ouvrir le groupe",
    callGroup: "Appel vidéo de groupe",
    ownerOnlyGroup:
      "Seul le propriétaire de l’organisation peut créer le groupe de l’équipe.",
    noMembers: "Aucun autre membre actif pour le moment.",
    messages: "Messages",
    chooseConversation:
      "Choisissez un membre ou le groupe de l’équipe pour commencer.",
    messagePlaceholder: "Écrire un message...",
    send: "Envoyer",
    sending: "Envoi...",
    realtimeConnecting: "Connexion...",
    messagePending: "Envoi...",
    messageFailed: "Échec de l’envoi",
    startCall: "Démarrer l’appel vidéo",
    joinCall: "Rejoindre l’appel vidéo",
    joining: "Connexion...",
    starting: "Démarrage...",
    online: "En ligne",
    offline: "Hors ligne",
    in_call: "En appel",
    unavailableTitle: "Messages et appels indisponibles",
    unavailableDescription:
      "Cet espace est réservé aux membres actifs d’une organisation Business ou Enterprise.",
    loading: "Chargement des messages...",
    directMessage: "Message direct",
    groupChat: "Groupe",
    memberChat: "Conversation membre",
    noConversations: "Aucune conversation pour le moment.",
    noMessagesOrCalls: "Aucun message ni journal d’appel pour le moment.",
    creating: "Création...",
    opening: "Ouverture...",
    noGroupYet: "Aucun groupe pour le moment.",
  },
};


const FOCUS_REFRESH_DEBOUNCE_MS = 750;

function titleCase(value) {
  if (!value) return "—";

  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function getErrorMessage(error) {
  return (
    error?.payload?.detail?.message ||
    error?.payload?.detail?.error ||
    error?.payload?.error?.message ||
    error?.message ||
    "Request failed"
  );
}

async function fetchJson(path, options = {}) {
  const token = await getAccessToken();

  const response = await fetch(path, {
    ...options,
    credentials: "include",
    cache: "no-store",
    headers: {
      Accept: "application/json",
      Authorization: `Bearer ${token}`,
      ...(options.headers || {}),
    },
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(
      data?.detail?.message ||
        data?.detail?.error ||
        data?.error?.message ||
        data?.message ||
        "Request failed",
    );
  }

  return data;
}

function getMessageCallSessionId(message) {
  const metadata = message?.metadata;

  if (!metadata || typeof metadata !== "object") {
    return null;
  }

  return metadata.call_session_id || metadata.callSessionId || null;
}

function getOrganizationName(details, entitlement) {
  return (
    details?.organization?.name ||
    details?.name ||
    entitlement?.organization_name ||
    "Team"
  );
}

function getMemberEmail(member) {
  return (
    member?.email ||
    member?.profile?.email ||
    member?.user?.email ||
    member?.user_id ||
    "No email available"
  );
}

function getMemberName(member) {
  const explicitName =
    member?.name ||
    member?.full_name ||
    member?.fullName ||
    member?.display_name ||
    member?.displayName ||
    member?.profile?.name ||
    member?.user?.name;

  if (explicitName) {
    return explicitName;
  }

  const email = getMemberEmail(member);

  if (email && email.includes("@")) {
    return email.split("@")[0];
  }

  return "Team member";
}

function getMemberJoinedTime(member) {
  const value =
    member?.joined_at ||
    member?.joinedAt ||
    member?.created_at ||
    member?.createdAt ||
    member?.updated_at ||
    member?.updatedAt;

  const timestamp = value ? new Date(value).getTime() : 0;
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function getConversationMemberIds(conversation) {
  const candidates =
    conversation?.member_user_ids ||
    conversation?.memberUserIds ||
    conversation?.participant_user_ids ||
    conversation?.participantUserIds ||
    conversation?.members ||
    conversation?.participants ||
    conversation?.conversation_members ||
    [];

  if (!Array.isArray(candidates)) {
    return [];
  }

  return candidates
    .map((item) => {
      if (typeof item === "string") return item;
      return item?.user_id || item?.userId || item?.id || null;
    })
    .filter(Boolean);
}

function createClientMessageId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return `client:${crypto.randomUUID()}`;
  }

  return `client:${Date.now()}:${Math.random().toString(16).slice(2)}`;
}

function getMessageClientId(message) {
  return (
    message?.client_message_id ||
    message?.clientMessageId ||
    message?.metadata?.client_message_id ||
    message?.metadata?.clientMessageId ||
    ""
  );
}

function buildOptimisticTextMessage({
  conversationId,
  organizationId,
  currentUserId,
  body,
  clientMessageId,
}) {
  const now = new Date().toISOString();

  return {
    id: clientMessageId,
    client_message_id: clientMessageId,
    conversation_id: conversationId,
    organization_id: organizationId,
    sender_user_id: currentUserId,
    message_type: "text",
    body,
    metadata: {
      client_message_id: clientMessageId,
      transport: "websocket",
      pending: true,
    },
    edited_at: null,
    deleted_at: null,
    created_at: now,
    updated_at: now,
    pending: true,
  };
}

export default function TeamMessagesPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { language } = useLanguage();
  const { sendRealtimeMessage, realtimeReady } = useTeamRealtime();
  const {
    user,
    entitlement,
    authChecked,
    loading: accountLoading,
  } = useAccount();
  const t = copy[language] || copy.en;
  const routeConversationId = useMemo(() => {
    const rawConversationId = searchParams.get("conversationId");
    const parsedConversationId = Number.parseInt(rawConversationId || "", 10);

    return Number.isFinite(parsedConversationId) && parsedConversationId > 0
      ? parsedConversationId
      : null;
  }, [searchParams]);
  const routeMessageId = searchParams.get("messageId") || "";
  const routeCallSessionId = searchParams.get("callSessionId") || "";

  const [loading, setLoading] = useState(true);
  const [organizationDetails, setOrganizationDetails] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [selectedConversationId, setSelectedConversationId] = useState(null);
  const selectedConversationIdRef = useRef(null);
  const refreshInFlightRef = useRef(false);
  const conversationSelectionRequestRef = useRef(0);
  const autoJoinedCallSessionRef = useRef(null);
  const [messages, setMessages] = useState([]);
  const [presence, setPresence] = useState([]);
  const [messageDraft, setMessageDraft] = useState("");
  const [activeCall, setActiveCall] = useState(null);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [highlightMessageId, setHighlightMessageId] = useState(null);

  const organizationId = entitlement?.organization_id || null;
  const isBusinessOrEnterprise =
    entitlement?.source === "organization" &&
    entitlement?.status === "active" &&
    ["business", "enterprise"].includes(entitlement?.plan);
  const currentUserId = user?.id;
  const isOwner = entitlement?.organization_role === "owner";

  const activeMembers = useMemo(
    () =>
      (organizationDetails?.members || [])
        .filter((member) => member.status === "active")
        .sort((a, b) => getMemberJoinedTime(b) - getMemberJoinedTime(a)),
    [organizationDetails],
  );

  const otherMembers = useMemo(
    () => activeMembers.filter((member) => member.user_id !== currentUserId),
    [activeMembers, currentUserId],
  );

  const memberByUserId = useMemo(
    () => new Map(activeMembers.map((member) => [member.user_id, member])),
    [activeMembers],
  );

  const presenceByUserId = useMemo(
    () => new Map(presence.map((entry) => [entry.user_id, entry])),
    [presence],
  );

  const selectedConversation = useMemo(
    () =>
      conversations.find(
        (conversation) => conversation.id === selectedConversationId,
      ) || null,
    [conversations, selectedConversationId],
  );

  const organizationName = useMemo(
    () => getOrganizationName(organizationDetails, entitlement),
    [organizationDetails, entitlement],
  );

  const groupConversation = useMemo(
    () =>
      conversations
        .filter((conversation) => conversation.type === "group")
        .sort((a, b) => {
          const aTime = new Date(a.updated_at || a.created_at || 0).getTime();
          const bTime = new Date(b.updated_at || b.created_at || 0).getTime();
          return bTime - aTime;
        })[0] || null,
    [conversations],
  );

  function getMemberLabel(userId) {
    const member = memberByUserId.get(userId);
    return member ? getMemberName(member) : userId;
  }

  function getConversationTitle(conversation) {
    if (!conversation) return "—";
    if (conversation.name) return conversation.name;

    if (conversation.type === "dm") {
      const otherParticipantId = getConversationMemberIds(conversation).find(
        (userId) => userId !== currentUserId,
      );

      return otherParticipantId
        ? getMemberLabel(otherParticipantId)
        : t.directMessage;
    }

    return `${organizationName} ${t.groupChat}`;
  }

  function getMemberPresenceStatus(userId) {
    return presenceByUserId.get(userId)?.status || "offline";
  }

  function getPresenceBadgeClass(status) {
    if (status === "in_call") {
      return "border-purple-400/30 bg-purple-400/10 text-purple-200";
    }

    if (status === "online") {
      return "border-emerald-400/30 bg-emerald-400/10 text-emerald-200";
    }

    return "border-[var(--app-border)] app-text-soft";
  }

  async function loadOrganizationDetails(nextOrganizationId) {
    const data = await fetchJson(`/api/organizations/${nextOrganizationId}`);
    setOrganizationDetails(data);
    return data;
  }

  async function loadConversations(
    nextOrganizationId,
    preferredConversationId,
    { selectFallback = false } = {},
  ) {
    const data = await getOrganizationConversations(nextOrganizationId);
    const nextConversations = data.conversations || [];

    setConversations(nextConversations);

    const currentSelectedId = selectedConversationIdRef.current;
    const nextSelected =
      nextConversations.find(
        (conversation) => conversation.id === preferredConversationId,
      ) ||
      nextConversations.find(
        (conversation) => conversation.id === currentSelectedId,
      ) ||
      (selectFallback
        ? nextConversations.find((conversation) => conversation.type === "group") ||
          nextConversations[0] ||
          null
        : null);

    if (nextSelected) {
      selectedConversationIdRef.current = nextSelected.id;
      setSelectedConversationId(nextSelected.id);
      return nextSelected;
    }

    if (selectFallback) {
      selectedConversationIdRef.current = null;
      setSelectedConversationId(null);
    }

    return null;
  }

  async function loadPresence(nextOrganizationId) {
    const data = await getOrganizationPresence(nextOrganizationId);
    setPresence(data.presence || []);
    return data.presence || [];
  }

  async function loadMessages(conversationId) {
    if (!conversationId) {
      setMessages([]);
      return [];
    }

    const data = await getConversationMessages(conversationId, { limit: 75 });
    setMessages(data.messages || []);
    return data.messages || [];
  }

  function upsertConversation(nextConversation) {
    if (!nextConversation?.id) return;

    setConversations((current) => {
      const exists = current.some(
        (conversation) => conversation.id === nextConversation.id,
      );
      const nextConversations = exists
        ? current.map((conversation) =>
            conversation.id === nextConversation.id
              ? { ...conversation, ...nextConversation }
              : conversation,
          )
        : [nextConversation, ...current];

      return [...nextConversations].sort((a, b) => {
        const aTime = new Date(
          a.last_message_at || a.updated_at || a.created_at || 0,
        ).getTime();
        const bTime = new Date(
          b.last_message_at || b.updated_at || b.created_at || 0,
        ).getTime();
        return bTime - aTime;
      });
    });
  }

  function upsertMessage(nextMessage, clientMessageId = "") {
    if (!nextMessage?.id) return;

    const currentSelectedId = selectedConversationIdRef.current;
    if (nextMessage.conversation_id !== currentSelectedId) return;

    const nextClientMessageId =
      clientMessageId || getMessageClientId(nextMessage) || "";

    const normalizedMessage = {
      ...nextMessage,
      client_message_id: nextClientMessageId || nextMessage.client_message_id,
      pending: Boolean(nextMessage.pending),
      failed: false,
    };

    setMessages((current) => {
      const nextMessageId = String(normalizedMessage.id);
      const existingIndex = current.findIndex((message) => {
        const currentMessageId = String(message.id);
        const currentClientMessageId = getMessageClientId(message);

        return (
          currentMessageId === nextMessageId ||
          (nextClientMessageId && currentClientMessageId === nextClientMessageId)
        );
      });

      const nextMessages = [...current];
      if (existingIndex >= 0) {
        nextMessages[existingIndex] = {
          ...nextMessages[existingIndex],
          ...normalizedMessage,
        };
      } else {
        nextMessages.push(normalizedMessage);
      }

      return nextMessages.sort((a, b) => {
        const aTime = new Date(a.created_at || 0).getTime();
        const bTime = new Date(b.created_at || 0).getTime();

        if (aTime !== bTime) return aTime - bTime;

        const aId = Number(a.id);
        const bId = Number(b.id);
        if (Number.isFinite(aId) && Number.isFinite(bId)) {
          return aId - bId;
        }

        return String(a.id || "").localeCompare(String(b.id || ""));
      });
    });
  }

  function markMessageFailed(clientMessageId, errorMessage) {
    if (!clientMessageId) return;

    setMessages((current) =>
      current.map((message) => {
        const currentClientMessageId = getMessageClientId(message);
        const currentMessageId = String(message.id || "");

        if (
          currentMessageId !== clientMessageId &&
          currentClientMessageId !== clientMessageId
        ) {
          return message;
        }

        return {
          ...message,
          pending: false,
          failed: true,
          error: errorMessage || t.messageFailed,
        };
      }),
    );
  }

  function upsertPresence(nextPresence) {
    if (!nextPresence?.user_id) return;

    setPresence((current) => {
      const exists = current.some(
        (entry) => entry.user_id === nextPresence.user_id,
      );

      if (!exists) return [...current, nextPresence];

      return current.map((entry) =>
        entry.user_id === nextPresence.user_id
          ? { ...entry, ...nextPresence }
          : entry,
      );
    });
  }

  function handleRealtimeEvent(event) {
    if (!event || event.organization_id !== organizationId) return;

    if (event.conversation) {
      upsertConversation(event.conversation);
    }

    if (event.presence) {
      upsertPresence(event.presence);
    }

    if (event.type === "conversation.created") {
      if (event.conversation?.id === selectedConversationIdRef.current) {
        void loadMessages(event.conversation.id);
      }
      return;
    }

    if (["message.created", "message.persisted", "message.ack"].includes(event.type)) {
      upsertMessage(
        {
          ...event.message,
          pending: event.type === "message.created" && Boolean(event.message?.pending),
        },
        event.client_message_id,
      );
      return;
    }

    if (event.type === "message.failed") {
      markMessageFailed(event.client_message_id, event.message);
      setNotice(event.message || t.messageFailed);
      return;
    }

    if (event.type === "call.started") {
      upsertMessage(event.message);
      return;
    }

    if (["call.joined", "call.left", "call.declined"].includes(event.type)) {
      if (event.call?.conversation_id === selectedConversationIdRef.current) {
        void loadMessages(event.call.conversation_id);
      }
    }
  }

  async function loadAll({ preferredConversationId } = {}) {
    if (!organizationId || !isBusinessOrEnterprise) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setNotice("");

    try {
      const [, nextSelected] = await Promise.all([
        loadOrganizationDetails(organizationId),
        loadConversations(organizationId, preferredConversationId, {
          selectFallback: true,
        }),
      ]);

      await Promise.all([
        loadPresence(organizationId),
        loadMessages(nextSelected?.id),
      ]);
    } catch (error) {
      setNotice(getErrorMessage(error));
    } finally {
      setLoading(false);
    }
  }

  async function refreshCurrentConversation() {
    if (!organizationId || refreshInFlightRef.current) return;

    const conversationId = selectedConversationIdRef.current;
    refreshInFlightRef.current = true;

    try {
      const tasks = [
        loadConversations(organizationId, conversationId, {
          selectFallback: false,
        }),
        loadPresence(organizationId),
      ];

      if (conversationId) {
        tasks.push(loadMessages(conversationId));
      }

      await Promise.all(tasks);
    } catch (error) {
      setNotice(getErrorMessage(error));
    } finally {
      refreshInFlightRef.current = false;
    }
  }

  async function selectConversation(conversationId) {
    const nextConversationId = conversationId || null;
    selectedConversationIdRef.current = nextConversationId;
    setSelectedConversationId(nextConversationId);
    setNotice("");
    await loadMessages(nextConversationId);
  }

  async function ensureDmConversation(member) {
    if (!organizationId || !member?.user_id || member.user_id === currentUserId) {
      return null;
    }

    const existing = conversations.find((conversation) => {
      if (conversation.type !== "dm") return false;

      const ids = getConversationMemberIds(conversation);
      return ids.includes(member.user_id) && ids.includes(currentUserId);
    });

    if (existing) {
      return existing;
    }

    const data = await createConversation(organizationId, {
      type: "dm",
      member_user_ids: [member.user_id],
    });

    await loadConversations(organizationId, data.conversation?.id, {
      selectFallback: false,
    });
    return data.conversation || null;
  }

  async function handleMessageMember(member) {
    const requestId = conversationSelectionRequestRef.current + 1;
    conversationSelectionRequestRef.current = requestId;

    setBusy(`message:${member.user_id}`);
    setNotice("");

    try {
      const conversation = await ensureDmConversation(member);

      if (conversation?.id && conversationSelectionRequestRef.current === requestId) {
        await selectConversation(conversation.id);
      }
    } catch (error) {
      setNotice(getErrorMessage(error));
    } finally {
      setBusy("");
    }
  }

  async function startCallForConversation(
    conversationId,
    { busyKey = `start-call:${conversationId}` } = {},
  ) {
    if (!conversationId) return;

    selectedConversationIdRef.current = conversationId;
    setSelectedConversationId(conversationId);
    setBusy(busyKey);
    setNotice("");

    try {
      const call = await startConversationCall(conversationId);
      setActiveCall(call);
      await Promise.all([
        loadMessages(conversationId),
        organizationId ? loadPresence(organizationId) : Promise.resolve(),
      ]);
    } catch (error) {
      setNotice(getErrorMessage(error));
    } finally {
      setBusy("");
    }
  }

  async function handleCallMember(member) {
    setBusy(`call:${member.user_id}`);
    setNotice("");

    try {
      const conversation = await ensureDmConversation(member);

      if (conversation?.id) {
        await startCallForConversation(conversation.id, {
          busyKey: `call:${member.user_id}`,
        });
      }
    } catch (error) {
      setNotice(getErrorMessage(error));
      setBusy("");
    }
  }

  async function handleCreateGroupConversation() {
    if (!organizationId || !isOwner || groupConversation) {
      return;
    }

    setBusy("create-group");
    setNotice("");

    try {
      const data = await createConversation(organizationId, {
        type: "group",
        name: `${organizationName} Team Chat`,
        member_user_ids: otherMembers.map((member) => member.user_id),
      });

      await loadConversations(organizationId, data.conversation?.id, {
        selectFallback: false,
      });
      await selectConversation(data.conversation?.id);
    } catch (error) {
      setNotice(getErrorMessage(error));
    } finally {
      setBusy("");
    }
  }

  async function handleOpenGroupConversation() {
    if (!groupConversation?.id) return;
    conversationSelectionRequestRef.current += 1;
    await selectConversation(groupConversation.id);
  }

  async function handleStartCurrentConversationCall() {
    await startCallForConversation(selectedConversationId);
  }

  async function handleStartGroupCall() {
    if (!groupConversation?.id) return;
    await startCallForConversation(groupConversation.id);
  }

  async function handleJoinCall(callSessionId) {
    if (!callSessionId) return;

    setBusy(`join-call:${callSessionId}`);
    setNotice("");

    try {
      const call = await joinCall(callSessionId);
      setActiveCall(call);
      if (organizationId) await loadPresence(organizationId);
    } catch (error) {
      setNotice(getErrorMessage(error));
    } finally {
      setBusy("");
    }
  }

  async function handleLeaveCall() {
    if (!activeCall?.call?.id) {
      setActiveCall(null);
      return;
    }

    const callId = activeCall.call.id;
    setActiveCall(null);

    try {
      await leaveCall(callId);
      if (organizationId) await loadPresence(organizationId);
    } catch (error) {
      setNotice(getErrorMessage(error));
    }
  }

  async function handleSendMessage(event) {
    event.preventDefault();

    const trimmedDraft = messageDraft.trim();
    const conversationId = selectedConversationIdRef.current;

    if (!conversationId || !trimmedDraft) {
      return;
    }

    const clientMessageId = createClientMessageId();
    const optimisticMessage = buildOptimisticTextMessage({
      conversationId,
      organizationId,
      currentUserId,
      body: trimmedDraft,
      clientMessageId,
    });

    setNotice("");
    setMessageDraft("");
    setMessages((current) => [...current, optimisticMessage]);

    try {
      sendRealtimeMessage({
        conversationId,
        body: trimmedDraft,
        clientMessageId,
      });
    } catch (error) {
      setMessages((current) =>
        current.filter((message) => getMessageClientId(message) !== clientMessageId),
      );
      setMessageDraft(trimmedDraft);
      setNotice(getErrorMessage(error));
    }
  }



  useEffect(() => {
    selectedConversationIdRef.current = selectedConversationId;
  }, [selectedConversationId]);

  useEffect(() => {
    if (!organizationId || !isBusinessOrEnterprise) return undefined;

    const listener = (event) => {
      handleRealtimeEvent(event.detail);
    };

    window.addEventListener("team-realtime-event", listener);

    return () => {
      window.removeEventListener("team-realtime-event", listener);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [organizationId, isBusinessOrEnterprise, selectedConversationId]);

  useEffect(() => {
    if (accountLoading || !authChecked) {
      return;
    }

    if (!user) {
      setLoading(false);
      setOrganizationDetails(null);
      setConversations([]);
      setSelectedConversationId(null);
      setMessages([]);
      setPresence([]);
      setNotice("");
      return;
    }

    if (routeMessageId) {
      setHighlightMessageId(routeMessageId);
    } else if (routeCallSessionId) {
      setHighlightMessageId(null);
    }

    void loadAll({ preferredConversationId: routeConversationId || undefined });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    accountLoading,
    authChecked,
    user?.id,
    organizationId,
    isBusinessOrEnterprise,
    routeConversationId,
    routeMessageId,
    routeCallSessionId,
  ]);

  useEffect(() => {
    if (accountLoading || !authChecked || !user) return;
    if (!organizationId || !isBusinessOrEnterprise) return;

    const callSessionId = Number.parseInt(routeCallSessionId || "", 10);

    if (!Number.isFinite(callSessionId) || callSessionId <= 0) return;
    if (activeCall?.call?.id === callSessionId) return;
    if (autoJoinedCallSessionRef.current === callSessionId) return;

    autoJoinedCallSessionRef.current = callSessionId;
    void handleJoinCall(callSessionId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    accountLoading,
    authChecked,
    user?.id,
    organizationId,
    isBusinessOrEnterprise,
    routeCallSessionId,
    activeCall?.call?.id,
  ]);

  useEffect(() => {
    if (accountLoading || !authChecked || !user) return undefined;
    if (!organizationId || !isBusinessOrEnterprise) return undefined;

    let focusTimeoutId = null;

    const handleFocus = () => {
      if (focusTimeoutId) {
        window.clearTimeout(focusTimeoutId);
      }

      focusTimeoutId = window.setTimeout(() => {
        void refreshCurrentConversation();
      }, FOCUS_REFRESH_DEBOUNCE_MS);
    };

    window.addEventListener("focus", handleFocus);

    return () => {
      if (focusTimeoutId) {
        window.clearTimeout(focusTimeoutId);
      }
      window.removeEventListener("focus", handleFocus);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    accountLoading,
    authChecked,
    user?.id,
    organizationId,
    isBusinessOrEnterprise,
    selectedConversationId,
  ]);

  useEffect(() => {
    if (!highlightMessageId || !messages.length) return undefined;

    const timeoutId = window.setTimeout(() => {
      const target = document.getElementById(
        `team-message-${highlightMessageId}`,
      );
      target?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 80);

    return () => window.clearTimeout(timeoutId);
  }, [highlightMessageId, messages]);

  if (accountLoading || !authChecked || loading) {
    return (
      <main className="flex h-dvh overflow-hidden app-page px-3 py-3 md:px-4 md:py-4">
        <div className="mx-auto flex h-full w-full max-w-7xl items-center justify-center rounded-2xl border app-surface-strong p-6 app-text">
          {t.loading}
        </div>
      </main>
    );
  }

  if (!organizationId || !isBusinessOrEnterprise) {
    return (
      <main className="flex h-dvh overflow-hidden app-page px-3 py-3 md:px-4 md:py-4">
        <section className="mx-auto flex max-h-full w-full max-w-4xl flex-col justify-center rounded-2xl border app-surface-strong p-6">
          <button
            type="button"
            onClick={() => router.push("/team")}
            className="mb-6 inline-flex items-center gap-2 rounded-2xl border app-surface px-4 py-2 text-sm font-semibold app-text"
          >
            <ArrowLeft className="h-4 w-4" />
            {t.backToTeam}
          </button>
          <h1 className="text-3xl font-semibold app-text">
            {t.unavailableTitle}
          </h1>
          <p className="mt-3 text-sm app-text-muted">
            {t.unavailableDescription}
          </p>
        </section>
      </main>
    );
  }

  return (
    <main className="h-dvh overflow-hidden app-page px-3 py-3 md:px-4 md:py-4">
      <div className="mx-auto flex h-full max-w-[1500px] flex-col gap-3 overflow-hidden">
        <header className="flex shrink-0 flex-col gap-3 rounded-2xl border app-surface-strong p-3 md:flex-row md:items-center md:justify-between">
          <div>
            <button
              type="button"
              onClick={() => router.push("/team")}
              className="mb-2 inline-flex items-center gap-2 rounded-xl border app-surface px-3 py-1.5 text-xs font-semibold app-text-soft transition hover:text-[var(--app-text)]"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              {t.backToTeam}
            </button>
            <h1 className="text-2xl font-semibold tracking-tight app-text md:text-3xl">
              {t.title}
            </h1>
            <p className="mt-1 max-w-3xl text-xs app-text-muted md:text-sm">
              {t.subtitle}
            </p>
          </div>
          <button
            type="button"
            onClick={() =>
              loadAll({ preferredConversationId: selectedConversationId })
            }
            className="inline-flex items-center justify-center gap-2 rounded-xl border app-surface px-3 py-2 text-sm font-semibold app-text transition hover:bg-[var(--app-button-bg)] hover:text-[var(--app-button-text)]"
          >
            <RefreshCw className="h-4 w-4" />
            {t.refresh}
          </button>
        </header>

        {notice ? (
          <div className="shrink-0 rounded-2xl border border-[var(--app-border)] app-surface-strong px-3 py-2 text-sm app-text">
            {notice}
          </div>
        ) : null}

        {activeCall ? (
          <div className="max-h-[40dvh] shrink-0 overflow-hidden rounded-2xl">
            <TeamCallRoom
              serverUrl={activeCall.livekit?.server_url}
              token={activeCall.livekit?.token}
              roomName={activeCall.livekit?.room_name}
              onLeave={handleLeaveCall}
            />
          </div>
        ) : null}

        <section className="grid min-h-0 flex-1 gap-3 lg:grid-cols-[17rem_minmax(0,1fr)_15rem] xl:grid-cols-[18rem_minmax(0,1fr)_16rem]">
          <aside className="min-h-0 overflow-hidden">
            <div className="flex h-full min-h-0 flex-col rounded-2xl border app-surface-strong p-3">
              <div className="mb-3 shrink-0">
                <h2 className="flex items-center gap-2 text-base font-semibold app-text">
                  <UsersRound className="h-5 w-5 app-text-muted" />
                  {t.teamMembers}
                </h2>
              </div>

              <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
                {otherMembers.length ? (
                  otherMembers.map((member, index) => {
                    const status = getMemberPresenceStatus(member.user_id);

                    return (
                      <div
                        key={member.user_id}
                        className="rounded-xl border app-surface p-2.5 transition hover:bg-[var(--app-surface-strong)]"
                      >
                        <div className="flex items-center gap-3">
                          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border app-surface-strong text-sm font-semibold app-text">
                            {getMemberName(member).slice(0, 1).toUpperCase()}
                          </div>

                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-2">
                              <p className="truncate text-sm font-semibold app-text">
                                {getMemberName(member)}
                              </p>
                              {index === 0 && otherMembers.length > 1 ? (
                                <span className="hidden rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-200 sm:inline-flex">
                                  {t.recentlyJoined}
                                </span>
                              ) : null}
                            </div>
                            <p className="truncate text-xs app-text-muted">
                              {getMemberEmail(member)}
                            </p>
                            <div className="mt-1 flex items-center gap-2 text-[11px] app-text-soft">
                              <span>{titleCase(member.role)}</span>
                              <span>·</span>
                              <span
                                className={`rounded-full border px-2 py-0.5 ${getPresenceBadgeClass(
                                  status,
                                )}`}
                              >
                                {t[status] || titleCase(status)}
                              </span>
                            </div>
                          </div>
                        </div>

                        <div className="mt-2 grid grid-cols-2 gap-2">
                          <button
                              type="button"
                              onClick={() => handleMessageMember(member)}
                              disabled={busy === `message:${member.user_id}`}
                              className="inline-flex items-center justify-center gap-1.5 rounded-xl border app-surface-strong px-2 py-1.5 text-xs font-semibold app-text transition hover:bg-[var(--app-button-bg)] hover:text-[var(--app-button-text)] disabled:cursor-not-allowed disabled:opacity-50"
                            >
                              <MessageCircle className="h-3.5 w-3.5" />
                              {busy === `message:${member.user_id}`
                                ? t.opening
                                : t.message}
                          </button>

                          <button
                              type="button"
                              onClick={() => handleCallMember(member)}
                              disabled={busy === `call:${member.user_id}`}
                              className="inline-flex items-center justify-center gap-1.5 rounded-xl bg-[var(--app-button-bg)] px-2 py-1.5 text-xs font-semibold text-[var(--app-button-text)] transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50"
                            >
                              <Video className="h-3.5 w-3.5" />
                              {busy === `call:${member.user_id}`
                                ? t.starting
                                : t.videoCall}
                          </button>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <p className="rounded-2xl border app-surface p-4 text-sm app-text-muted">
                    {t.noMembers}
                  </p>
                )}
              </div>
            </div>
          </aside>

          <section className="flex min-h-0 flex-col rounded-2xl border app-surface-strong p-3">
            <div className="mb-3 flex shrink-0 flex-col gap-3 border-b border-[var(--app-border)] pb-3 md:flex-row md:items-center md:justify-between">
              <div className="min-w-0">
                <h2 className="truncate text-lg font-semibold app-text">
                  {selectedConversation
                    ? getConversationTitle(selectedConversation)
                    : t.messages}
                </h2>
                <p className="mt-1 text-xs app-text-soft">
                  {selectedConversation
                    ? selectedConversation.type === "dm"
                      ? t.directMessage
                      : t.groupChat
                    : t.chooseConversation}
                </p>
              </div>

              {selectedConversation ? (
                <button
                  type="button"
                  onClick={handleStartCurrentConversationCall}
                  disabled={busy === `start-call:${selectedConversationId}`}
                  className="inline-flex items-center justify-center gap-2 rounded-xl border app-surface px-3 py-2 text-sm font-semibold app-text transition hover:bg-[var(--app-button-bg)] hover:text-[var(--app-button-text)] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Video className="h-4 w-4" />
                  {busy === `start-call:${selectedConversationId}`
                    ? t.starting
                    : t.startCall}
                </button>
              ) : null}
            </div>

            <div className="flex min-h-0 flex-1 flex-col">
              <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-2">
                {selectedConversation && messages.length ? (
                  messages.map((message) => {
                    const isMine = message.sender_user_id === currentUserId;
                    const callSessionId = getMessageCallSessionId(message);
                    const isCallEvent = message.message_type === "call_event";
                    const isHighlighted =
                      highlightMessageId &&
                      String(message.id) === String(highlightMessageId);

                    return (
                      <div
                        id={`team-message-${message.id}`}
                        key={message.id}
                        className={`flex scroll-mt-24 ${isMine ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`max-w-[85%] rounded-2xl border px-3.5 py-2.5 text-sm transition ${
                            isHighlighted
                              ? "ring-2 ring-[var(--app-button-bg)] ring-offset-2 ring-offset-[var(--app-bg)]"
                              : ""
                          } ${
                            isMine
                              ? "bg-[var(--app-button-bg)] text-[var(--app-button-text)]"
                              : "app-surface app-text"
                          }`}
                        >
                          <div
                            className={`mb-1 text-[11px] font-semibold ${
                              isMine ? "opacity-70" : "app-text-soft"
                            }`}
                          >
                            {isMine ? t.you : getMemberLabel(message.sender_user_id)}
                          </div>
                          <div className="whitespace-pre-wrap leading-6">
                            {message.body}
                          </div>
                          {isMine && (message.pending || message.failed) ? (
                            <div
                              className={`mt-2 text-[11px] font-semibold ${
                                message.failed
                                  ? "text-red-300"
                                  : isMine
                                    ? "opacity-70"
                                    : "app-text-soft"
                              }`}
                            >
                              {message.failed
                                ? message.error || t.messageFailed
                                : t.messagePending}
                            </div>
                          ) : null}
                          {isCallEvent && callSessionId ? (
                            <button
                              type="button"
                              onClick={() => handleJoinCall(callSessionId)}
                              disabled={busy === `join-call:${callSessionId}`}
                              className={`mt-3 inline-flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-semibold transition ${
                                isMine
                                  ? "border-black/20 text-black"
                                  : "app-surface app-text"
                              }`}
                            >
                              <Video className="h-3.5 w-3.5" />
                              {busy === `join-call:${callSessionId}`
                                ? t.joining
                                : t.joinCall}
                            </button>
                          ) : null}
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="rounded-xl border app-surface p-4 text-sm app-text-muted">
                    {selectedConversation ? t.noMessagesOrCalls : t.chooseConversation}
                  </div>
                )}
              </div>

              <form onSubmit={handleSendMessage} className="mt-3 flex shrink-0 gap-2">
                <input
                  type="text"
                  value={messageDraft}
                  onChange={(event) => setMessageDraft(event.target.value)}
                  placeholder={t.messagePlaceholder}
                  disabled={!selectedConversation}
                  className="min-w-0 flex-1 rounded-xl border px-4 py-2.5 text-sm"
                />
                <button
                  type="submit"
                  disabled={
                    !selectedConversation ||
                    !messageDraft.trim() ||
                    !realtimeReady
                  }
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-[var(--app-button-bg)] px-4 py-2.5 text-sm font-semibold text-[var(--app-button-text)] transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Send className="h-4 w-4" />
                  <span className="hidden sm:inline">
                    {!realtimeReady ? t.realtimeConnecting : t.send}
                  </span>
                </button>
              </form>
            </div>
          </section>

          <aside className="min-h-0 overflow-hidden">
            <div className="flex h-full min-h-0 flex-col rounded-2xl border app-surface-strong p-3">
              <h2 className="mb-2 flex shrink-0 items-center gap-2 text-base font-semibold app-text">
                <UsersRound className="h-5 w-5 app-text-muted" />
                {t.groupWorkspace}
              </h2>
              <div className="mt-3 space-y-2 overflow-y-auto pr-1">
                {groupConversation ? (
                  <>
                    <button
                      type="button"
                      onClick={handleOpenGroupConversation}
                      className="inline-flex w-full items-center justify-center gap-2 rounded-xl border app-surface px-3 py-2.5 text-sm font-semibold app-text transition hover:bg-[var(--app-button-bg)] hover:text-[var(--app-button-text)]"
                    >
                      <MessageCircle className="h-4 w-4" />
                      {t.openGroupChat}
                    </button>

                    <button
                      type="button"
                      onClick={handleStartGroupCall}
                      disabled={busy === `start-call:${groupConversation.id}`}
                      className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-[var(--app-button-bg)] px-3 py-2.5 text-sm font-semibold text-[var(--app-button-text)] transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <Video className="h-4 w-4" />
                      {busy === `start-call:${groupConversation.id}`
                        ? t.starting
                        : t.callGroup}
                    </button>
                  </>
                ) : (
                  <>
                    <div className="rounded-xl border app-surface px-3 py-2.5 text-sm app-text-muted">
                      {isOwner ? t.noGroupYet : t.ownerOnlyGroup}
                    </div>

                    {isOwner ? (
                      <button
                        type="button"
                        onClick={handleCreateGroupConversation}
                        disabled={busy === "create-group"}
                        className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-[var(--app-button-bg)] px-3 py-2.5 text-sm font-semibold text-[var(--app-button-text)] transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        <UsersRound className="h-4 w-4" />
                        {busy === "create-group"
                          ? t.creating
                          : t.createGroupChat}
                      </button>
                    ) : null}
                  </>
                )}
              </div>
            </div>

          </aside>
        </section>
      </div>
    </main>
  );
}
