export interface ElectronAPI {
  predictHatespeech: (text: string) => Promise<{
    success: boolean;
    data?: {
      label: string;
      probability: number;
      probabilities: {
        non_hate: number;
        hate: number;
      };
    };
    error?: string;
  }>;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

