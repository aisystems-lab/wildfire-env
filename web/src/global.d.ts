// So we can import CSS modules.
declare module "*.sass";
declare module "*.scss";
declare module "*.png" {
  const value: string;
  export = value;
}

declare const __APP_CONFIG__: {
  wsUrl: string;
  generatedAssetsBaseUrl: string;
};
