declare const __APP_CONFIG__: {
  wsUrl: string;
  generatedAssetsBaseUrl: string;
};

const defaultWsUrl = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws`;

export const WS_URL = (__APP_CONFIG__.wsUrl || defaultWsUrl).replace(/\/$/, "");
export const GENERATED_ASSETS_BASE_URL = __APP_CONFIG__.generatedAssetsBaseUrl.replace(/\/$/, "");
