"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { getAccountMe } from "@/lib/api_client";

const AccountContext = createContext({
  account: null,
  user: null,
  settings: null,
  entitlement: null,
  authChecked: false,
  isSignedIn: false,
  loading: true,
  error: null,
  reloadAccount: async () => null,
});

export function AccountProvider({ children }) {
  const [account, setAccount] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadAccount = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await getAccountMe();
      setAccount(data);
      setAuthChecked(true);
      return data;
    } catch (caught) {
      const status = caught?.status;

      if (status === 401 || status === 403) {
        setAccount(null);
        setAuthChecked(true);
        return null;
      }

      setAccount(null);
      setAuthChecked(true);
      setError(caught instanceof Error ? caught : new Error("Could not load account."));
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;

    async function run() {
      const result = await loadAccount();
      if (!active) return result;
      return result;
    }

    void run();

    return () => {
      active = false;
    };
  }, [loadAccount]);

  const value = useMemo(
    () => ({
      account,
      user: account?.user || null,
      settings: account?.settings || null,
      entitlement: account?.entitlement || null,
      authChecked,
      isSignedIn: !!account?.user,
      loading,
      error,
      reloadAccount: loadAccount,
    }),
    [account, authChecked, error, loadAccount, loading],
  );

  return (
    <AccountContext.Provider value={value}>{children}</AccountContext.Provider>
  );
}

export function useAccount() {
  return useContext(AccountContext);
}
