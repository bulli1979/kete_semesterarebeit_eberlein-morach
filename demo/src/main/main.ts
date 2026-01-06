import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';
import { predict, loadModel } from './model';

let mainWindow: BrowserWindow | null = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, '../preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  // Prüfe ob Vite Dev Server läuft, sonst lade statische Dateien
  const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
  
  // Öffne DevTools immer, um Fehler zu sehen
  mainWindow.webContents.openDevTools();
  
  if (isDev) {
    // Versuche Vite Dev Server zu laden
    mainWindow.loadURL('http://localhost:3000').catch((err) => {
      console.error('Fehler beim Laden des Dev Servers:', err);
      // Falls Dev Server nicht läuft, lade statische Dateien
      const rendererPath = path.join(__dirname, '../renderer');
      console.log('Lade statische Dateien von:', rendererPath);
      mainWindow?.loadFile(path.join(rendererPath, 'index.html'));
    });
  } else {
    // In Produktion: Statische Dateien
    const rendererPath = path.join(__dirname, '../renderer');
    console.log('Lade statische Dateien von:', rendererPath);
    mainWindow.loadFile(path.join(rendererPath, 'index.html'));
  }

  // Fehlerbehandlung für fehlgeschlagene Ladungen
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription, validatedURL) => {
    console.error('Fehler beim Laden:', errorCode, errorDescription, validatedURL);
    // Falls Dev Server nicht verfügbar, versuche statische Dateien
    if (isDev && errorCode === -106) {
      const rendererPath = path.join(__dirname, '../renderer');
      console.log('Fallback: Lade statische Dateien von:', rendererPath);
      mainWindow?.loadFile(path.join(rendererPath, 'index.html'));
    }
  });
  
  // Logge alle Console-Nachrichten
  mainWindow.webContents.on('console-message', (event, level, message) => {
    console.log(`[Renderer ${level}]:`, message);
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  // Erstelle Fenster zuerst, damit UI sofort sichtbar ist
  createWindow();

  // Lade Modell im Hintergrund (nicht blockierend)
  loadModel()
    .then(() => {
      console.log('✅ Modell geladen');
    })
    .catch((error) => {
      console.error('❌ Fehler beim Laden des Modells:', error);
      // Zeige Fehler im Fenster, falls es bereits geöffnet ist
      if (mainWindow) {
        mainWindow.webContents.executeJavaScript(`
          console.error('Modell konnte nicht geladen werden:', ${JSON.stringify(error.message)});
        `);
      }
    });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC Handler für Hatespeech-Erkennung
ipcMain.handle('predict-hatespeech', async (_event, text: string) => {
  try {
    const result = await predict(text);
    return { success: true, data: result };
  } catch (error: any) {
    console.error('Fehler bei Vorhersage:', error);
    return { success: false, error: error.message };
  }
});

