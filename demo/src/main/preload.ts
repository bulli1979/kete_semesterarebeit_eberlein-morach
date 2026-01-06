import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  predictHatespeech: (text: string) => ipcRenderer.invoke('predict-hatespeech', text),
});

