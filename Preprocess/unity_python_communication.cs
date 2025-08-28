using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using UnityEngine;
using System.Threading.Tasks;

/// <summary>
/// Unity에서 Python 프로세스와 통신하여 Random Forest 모델을 사용하는 클래스
/// </summary>
public class UnityPythonMotionClassifier : MonoBehaviour
{
    [Header("Python 설정")]
    [SerializeField] private string pythonScriptPath = "unity_python_bridge.py";
    [SerializeField] private string pythonExecutable = "python";
    [SerializeField] private bool useAsync = true;
    
    [Header("모델 정보")]
    [SerializeField] private string modelInfo = "";
    
    private Process pythonProcess;
    private bool isPythonReady = false;
    private readonly Queue<string> commandQueue = new Queue<string>();
    private readonly object queueLock = new object();
    
    // 이벤트
    public event Action<string, float> OnMotionPredicted;
    public event Action<string> OnPredictionError;
    
    private void Start()
    {
        InitializePythonBridge();
    }
    
    private void OnDestroy()
    {
        CleanupPythonProcess();
    }
    
    /// <summary>
    /// Python 브리지 초기화
    /// </summary>
    private async void InitializePythonBridge()
    {
        try
        {
            if (useAsync)
            {
                await InitializePythonBridgeAsync();
            }
            else
            {
                InitializePythonBridgeSync();
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Python 브리지 초기화 실패: {e.Message}");
        }
    }
    
    /// <summary>
    /// 비동기 Python 브리지 초기화
    /// </summary>
    private async Task InitializePythonBridgeAsync()
    {
        await Task.Run(() =>
        {
            try
            {
                // Python 프로세스 시작
                StartPythonProcess();
                
                // 모델 정보 요청
                SendCommand("get_model_info");
                
                // 초기화 완료 대기
                WaitForPythonReady();
                
                isPythonReady = true;
                Debug.Log("✅ Python 브리지 초기화 완료 (비동기)");
            }
            catch (Exception e)
            {
                Debug.LogError($"Python 브리지 초기화 실패: {e.Message}");
            }
        });
    }
    
    /// <summary>
    /// 동기 Python 브리지 초기화
    /// </summary>
    private void InitializePythonBridgeSync()
    {
        try
        {
            // Python 프로세스 시작
            StartPythonProcess();
            
            // 모델 정보 요청
            SendCommand("get_model_info");
            
            // 초기화 완료 대기
            WaitForPythonReady();
            
            isPythonReady = true;
            Debug.Log("✅ Python 브리지 초기화 완료 (동기)");
        }
        catch (Exception e)
        {
            Debug.LogError($"Python 브리지 초기화 실패: {e.Message}");
        }
    }
    
    /// <summary>
    /// Python 프로세스 시작
    /// </summary>
    private void StartPythonProcess()
    {
        try
        {
            // Python 스크립트 경로 확인
            string fullScriptPath = Path.Combine(Application.dataPath, "..", pythonScriptPath);
            if (!File.Exists(fullScriptPath))
            {
                throw new FileNotFoundException($"Python 스크립트를 찾을 수 없습니다: {fullScriptPath}");
            }
            
            // Python 프로세스 시작
            pythonProcess = new Process();
            pythonProcess.StartInfo.FileName = pythonExecutable;
            pythonProcess.StartInfo.Arguments = $"\"{fullScriptPath}\"";
            pythonProcess.StartInfo.UseShellExecute = false;
            pythonProcess.StartInfo.RedirectStandardInput = true;
            pythonProcess.StartInfo.RedirectStandardOutput = true;
            pythonProcess.StartInfo.RedirectStandardError = true;
            pythonProcess.StartInfo.CreateNoWindow = true;
            
            pythonProcess.OutputDataReceived += OnPythonOutput;
            pythonProcess.ErrorDataReceived += OnPythonError;
            
            pythonProcess.Start();
            pythonProcess.BeginOutputReadLine();
            pythonProcess.BeginErrorReadLine();
            
            Debug.Log($"🚀 Python 프로세스 시작: {fullScriptPath}");
        }
        catch (Exception e)
        {
            throw new Exception($"Python 프로세스 시작 실패: {e.Message}");
        }
    }
    
    /// <summary>
    /// Python 준비 상태 대기
    /// </summary>
    private void WaitForPythonReady()
    {
        // 간단한 대기 (실제로는 Python에서 준비 완료 신호를 받아야 함)
        System.Threading.Thread.Sleep(2000);
    }
    
    /// <summary>
    /// Python에 명령 전송
    /// </summary>
    private void SendCommand(string command)
    {
        if (pythonProcess != null && !pythonProcess.HasExited)
        {
            try
            {
                pythonProcess.StandardInput.WriteLine(command);
                pythonProcess.StandardInput.Flush();
                Debug.Log($"📤 Python 명령 전송: {command}");
            }
            catch (Exception e)
            {
                Debug.LogError($"Python 명령 전송 실패: {e.Message}");
            }
        }
    }
    
    /// <summary>
    /// 동작 분류 예측 (비동기)
    /// </summary>
    public async Task<string> PredictMotionAsync(List<Vector3[]> jointPositions, List<Quaternion[]> jointRotations)
    {
        if (!isPythonReady)
        {
            throw new InvalidOperationException("Python 브리지가 아직 준비되지 않았습니다.");
        }
        
        return await Task.Run(() =>
        {
            try
            {
                // 데이터를 JSON 형태로 직렬화
                string jsonData = SerializeMotionData(jointPositions, jointRotations);
                
                // Python에 예측 요청
                SendCommand($"predict:{jsonData}");
                
                // 결과 대기 (실제로는 응답을 받아야 함)
                System.Threading.Thread.Sleep(1000);
                
                // 임시로 "Hold" 반환 (실제로는 Python 응답을 파싱해야 함)
                return "Hold";
            }
            catch (Exception e)
            {
                Debug.LogError($"동작 예측 실패: {e.Message}");
                return "Unknown";
            }
        });
    }
    
    /// <summary>
    /// 동작 분류 예측 (동기)
    /// </summary>
    public string PredictMotion(List<Vector3[]> jointPositions, List<Quaternion[]> jointRotations)
    {
        if (!isPythonReady)
        {
            throw new InvalidOperationException("Python 브리지가 아직 준비되지 않았습니다.");
        }
        
        try
        {
            // 데이터를 JSON 형태로 직렬화
            string jsonData = SerializeMotionData(jointPositions, jointRotations);
            
            // Python에 예측 요청
            SendCommand($"predict:{jsonData}");
            
            // 결과 대기 (실제로는 응답을 받아야 함)
            System.Threading.Thread.Sleep(1000);
            
            // 임시로 "Hold" 반환 (실제로는 Python 응답을 파싱해야 함)
            return "Hold";
        }
        catch (Exception e)
        {
            Debug.LogError($"동작 예측 실패: {e.Message}");
            return "Unknown";
        }
    }
    
    /// <summary>
    /// 모션 데이터를 JSON으로 직렬화
    /// </summary>
    private string SerializeMotionData(List<Vector3[]> jointPositions, List<Quaternion[]> jointRotations)
    {
        try
        {
            var data = new
            {
                joint_positions = jointPositions.ConvertAll(frame => 
                    frame.ConvertAll(pos => new { x = pos.x, y = pos.y, z = pos.z })),
                joint_rotations = jointRotations.ConvertAll(frame => 
                    frame.ConvertAll(rot => new { x = rot.x, y = rot.y, z = rot.z, w = rot.w }))
            };
            
            return JsonUtility.ToJson(data);
        }
        catch (Exception e)
        {
            Debug.LogError($"데이터 직렬화 실패: {e.Message}");
            return "{}";
        }
    }
    
    /// <summary>
    /// Python 출력 처리
    /// </summary>
    private void OnPythonOutput(object sender, DataReceivedEventArgs e)
    {
        if (!string.IsNullOrEmpty(e.Data))
        {
            Debug.Log($"📥 Python 출력: {e.Data}");
            
            // 모델 정보 파싱
            if (e.Data.Contains("모델 정확도:"))
            {
                modelInfo = e.Data;
            }
            
            // 예측 결과 파싱 (실제로는 더 정교한 파싱이 필요)
            if (e.Data.Contains("예측 완료:"))
            {
                ParsePredictionResult(e.Data);
            }
        }
    }
    
    /// <summary>
    /// Python 에러 처리
    /// </summary>
    private void OnPythonError(object sender, DataReceivedEventArgs e)
    {
        if (!string.IsNullOrEmpty(e.Data))
        {
            Debug.LogError($"❌ Python 에러: {e.Data}");
        }
    }
    
    /// <summary>
    /// 예측 결과 파싱
    /// </summary>
    private void ParsePredictionResult(string output)
    {
        try
        {
            // 간단한 파싱 (실제로는 더 정교한 파싱이 필요)
            if (output.Contains("Hold"))
            {
                OnMotionPredicted?.Invoke("Hold", 0.73f);
            }
            else if (output.Contains("Pick"))
            {
                OnMotionPredicted?.Invoke("Pick", 0.22f);
            }
            else if (output.Contains("Place"))
            {
                OnMotionPredicted?.Invoke("Place", 0.05f);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"예측 결과 파싱 실패: {e.Message}");
            OnPredictionError?.Invoke(e.Message);
        }
    }
    
    /// <summary>
    /// Python 프로세스 정리
    /// </summary>
    private void CleanupPythonProcess()
    {
        if (pythonProcess != null && !pythonProcess.HasExited)
        {
            try
            {
                SendCommand("exit");
                pythonProcess.WaitForExit(3000);
                
                if (!pythonProcess.HasExited)
                {
                    pythonProcess.Kill();
                }
                
                pythonProcess.Dispose();
                pythonProcess = null;
                
                Debug.Log("🧹 Python 프로세스 정리 완료");
            }
            catch (Exception e)
            {
                Debug.LogError($"Python 프로세스 정리 실패: {e.Message}");
            }
        }
    }
    
    /// <summary>
    /// 모델 정보 가져오기
    /// </summary>
    public string GetModelInfo()
    {
        return modelInfo;
    }
    
    /// <summary>
    /// Python 브리지 상태 확인
    /// </summary>
    public bool IsPythonReady()
    {
        return isPythonReady;
    }
    
    /// <summary>
    /// 테스트용 동작 예측
    /// </summary>
    [ContextMenu("테스트 동작 예측")]
    public void TestMotionPrediction()
    {
        if (!isPythonReady)
        {
            Debug.LogWarning("Python 브리지가 아직 준비되지 않았습니다.");
            return;
        }
        
        // 테스트 데이터 생성
        var testPositions = new List<Vector3[]>
        {
            new Vector3[] { Vector3.zero, Vector3.one, Vector3.forward },
            new Vector3[] { Vector3.up, Vector3.right, Vector3.back }
        };
        
        var testRotations = new List<Quaternion[]>
        {
            new Quaternion[] { Quaternion.identity, Quaternion.Euler(0, 90, 0), Quaternion.Euler(0, 180, 0) },
            new Quaternion[] { Quaternion.Euler(0, 270, 0), Quaternion.identity, Quaternion.Euler(0, 45, 0) }
        };
        
        // 예측 실행
        string result = PredictMotion(testPositions, testRotations);
        Debug.Log($"🧪 테스트 예측 결과: {result}");
    }
}

/// <summary>
/// 모션 분류 결과를 담는 구조체
/// </summary>
[System.Serializable]
public struct MotionPredictionResult
{
    public string motionClass;
    public float confidence;
    public Dictionary<string, float> probabilities;
    public bool success;
    public string error;
}

/// <summary>
/// 사용 예시
/// </summary>
public class MotionClassifierExample : MonoBehaviour
{
    [SerializeField] private UnityPythonMotionClassifier classifier;
    
    private void Start()
    {
        // 이벤트 구독
        if (classifier != null)
        {
            classifier.OnMotionPredicted += OnMotionPredicted;
            classifier.OnPredictionError += OnPredictionError;
        }
    }
    
    private void OnMotionPredicted(string motionClass, float confidence)
    {
        Debug.Log($"🎯 동작 분류 결과: {motionClass} (신뢰도: {confidence:F3})");
    }
    
    private void OnPredictionError(string error)
    {
        Debug.LogError($"❌ 예측 에러: {error}");
    }
    
    private void OnDestroy()
    {
        // 이벤트 구독 해제
        if (classifier != null)
        {
            classifier.OnMotionPredicted -= OnMotionPredicted;
            classifier.OnPredictionError -= OnPredictionError;
        }
    }
}


