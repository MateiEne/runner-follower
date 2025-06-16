using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.IO;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

public class TCPServer : MonoBehaviour
{
    public Camera targetCamera;
    public RenderTexture renderTexture;

    TcpListener _tcpListener;
    TcpClient _tcpClient;
    bool _clientConnected = false;

    private Queue<Action> _mainThreadQueue = new Queue<Action>(); // Coada pentru thread-ul principal
    
    RobotController _robotController;

    void Awake()
    {
        DontDestroyOnLoad(this);
        StartListening();
    }

    private void Start()
    {
        GameObject robot = GameObject.Find("Robot");
        _robotController = robot.GetComponent<RobotController>();
    }

    void StartListening()
    {
        IPAddress ipaddr;
        int nPort = 2737;

        IPAddress.TryParse("127.0.0.1", out ipaddr);
        _tcpListener = new TcpListener(ipaddr, nPort);
        _tcpListener.Start(1);

        Debug.Log("Server started... Waiting for client...");
        _tcpListener.BeginAcceptTcpClient(OnCompleteAcceptTcpClient, _tcpListener);
    }

    void Update()
    {
        // Execută toate acțiunile programate pentru thread-ul principal
        while (_mainThreadQueue.Count > 0)
        {
            _mainThreadQueue.Dequeue().Invoke();
        }
    }

    void OnCompleteAcceptTcpClient(IAsyncResult iar)
    {
        TcpListener tcpl = (TcpListener)iar.AsyncState;

        if (tcpl == null || !_tcpListener.Server.IsBound)  // Verifică dacă listenerul e activ
        {
            Debug.LogWarning("Server is shutting down, ignoring new connections.");
            return;
        }

        try
        {
            _tcpClient = tcpl.EndAcceptTcpClient(iar);
            _clientConnected = true;
            Debug.Log("Client connected.");

            // Acceptă următorul client doar dacă serverul e încă activ
            if (_tcpListener != null && _tcpListener.Server.IsBound)
            {
                tcpl.BeginAcceptTcpClient(OnCompleteAcceptTcpClient, tcpl);
            }

            // Începe trimiterea de imagini, dar doar dacă suntem conectați
            if (_clientConnected)
            {
                _mainThreadQueue.Enqueue(() => StartCoroutine(SendImagesContinuously()));
                _mainThreadQueue.Enqueue(() => StartCoroutine(ReceiveAndProcessData()));
            }
        }
        catch (ObjectDisposedException)
        {
            Debug.LogWarning("Server socket was closed. Stopping accept loop.");
        }
        catch (Exception ex)
        {
            Debug.LogError("Error accepting client: " + ex.Message);
            _clientConnected = false;
        }
    }

    IEnumerator SendImagesContinuously()
    {
        while (_clientConnected && _tcpClient != null)
        {
            SendImageToClient();
            yield return new WaitForSeconds(0.1f); // Trimite la fiecare 100ms (10 FPS)
            //yield return new WaitForSecondsRealtime(0.1f);
        }
    }

    IEnumerator ReceiveAndProcessData()
    {
        NetworkStream stream = _tcpClient.GetStream();
        byte[] sizeBytes = new byte[4];
        int bytesRead;

        while (_clientConnected && _tcpClient != null)
        {
            // STEP 1: Read size
            IAsyncResult readSizeResult = null;
            try
            {
                readSizeResult = stream.BeginRead(sizeBytes, 0, 4, null, null);
            }
            catch (Exception e)
            {
                Debug.LogError("BeginRead (size) failed: " + e.Message);
                _clientConnected = false;
                yield break;
            }

            yield return new WaitUntil(() => readSizeResult.IsCompleted);

            try
            {
                bytesRead = stream.EndRead(readSizeResult);
                if (bytesRead != 4)
                {
                    Debug.LogError("Failed to read message size. Disconnecting.");
                    _clientConnected = false;
                    yield break;
                }
            }
            catch (Exception e)
            {
                Debug.LogError("EndRead (size) failed: " + e.Message);
                _clientConnected = false;
                yield break;
            }

            // STEP 2: Read message content
            int messageSize = BitConverter.ToInt32(sizeBytes, 0);
            byte[] messageBytes = new byte[messageSize];
            int totalBytesRead = 0;

            while (totalBytesRead < messageSize)
            {
                IAsyncResult readMessageResult = null;
                try
                {
                    readMessageResult = stream.BeginRead(messageBytes, totalBytesRead, messageSize - totalBytesRead, null, null);
                }
                catch (Exception e)
                {
                    Debug.LogError("BeginRead (message) failed: " + e.Message);
                    _clientConnected = false;
                    yield break;
                }

                yield return new WaitUntil(() => readMessageResult.IsCompleted);

                try
                {
                    bytesRead = stream.EndRead(readMessageResult);
                    if (bytesRead == 0)
                    {
                        Debug.LogWarning("Client disconnected unexpectedly.");
                        _clientConnected = false;
                        yield break;
                    }
                    totalBytesRead += bytesRead;
                }
                catch (Exception e)
                {
                    Debug.LogError("EndRead (message) failed: " + e.Message);
                    _clientConnected = false;
                    yield break;
                }
            }

            // STEP 3: Process message
            string clientMessage = null;
            try
            {
                clientMessage = Encoding.UTF8.GetString(messageBytes);
                _robotController.ProcessCommand(clientMessage);
            }
            catch (Exception e)
            {
                Debug.LogError("Message decoding failed: " + e.Message);
                _clientConnected = false;
                yield break;
            }

            yield return null;
        }
    }

    void SendImageToClient()
    {
        if (!_clientConnected || _tcpClient == null)
        {
            Debug.LogWarning("No client connected!");
            return;
        }

        try
        {
            NetworkStream stream = _tcpClient.GetStream();

            if (targetCamera == null)
            {
                Debug.LogError("Target Camera is null!");
                return;
            }

            Texture2D texture = RenderTextureToTexture2D(renderTexture);
            byte[] imageBytes = texture.EncodeToJPG(75);

            if (imageBytes == null || imageBytes.Length == 0)
            {
                Debug.LogError("Image encoding failed!");
                return;
            }

            byte[] sizeInfo = BitConverter.GetBytes(imageBytes.Length);
            stream.Write(sizeInfo, 0, sizeInfo.Length);
            stream.Write(imageBytes, 0, imageBytes.Length);
            Destroy(texture);
        }
        catch (Exception ex)
        {
            Debug.LogError("Error sending image to client: " + ex.Message);
            _clientConnected = false;
        }
    }

    private Texture2D RenderTextureToTexture2D(RenderTexture target)
    {
        Texture2D result = new Texture2D(target.width, target.height, TextureFormat.RGB24, false);
        RenderTexture.active = target;
        result.ReadPixels(new Rect(0, 0, target.width, target.height), 0, 0);
        result.Apply();

        RenderTexture.active = null;

        return result;
    }

    private void OnDestroy()
    {
        _clientConnected = false; // Oprește trimiterea imaginilor

        if (_tcpListener != null)
        {
            _tcpListener.Stop();
            _tcpListener = null;
        }

        if (_tcpClient != null)
        {
            _tcpClient.Close();
            _tcpClient = null;
        }

        Debug.Log("Server shutdown completed.");
    }
}