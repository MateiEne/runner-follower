using UnityEngine;

public class RobotPhysicsController : MonoBehaviour
{
    Rigidbody _rBody;

    [Header("Motor settings")]
    public float maxRPM = 12000f;
    public float wheelRadius = 0.03f;
    public float maxSpeed = 1.5f; // m/s
    public float motorAcceleration = 5f;

    [Header("Steering settings")]
    public float maxSteeringAngle = 30f; // degrees
    public float steeringSpeed = 10f;
    public float maxSteeringCommand = 2f; // max PID output

    [HideInInspector]
    public float speedCommand = 0f; // from PID
    [HideInInspector]
    public float steeringCommand = 0f; // from PID

    float currentSpeed = 0f;
    float currentSteeringAngle = 0f;

    void Start()
    {
        _rBody = GetComponent<Rigidbody>();
        if (_rBody == null)
            Debug.LogError("Missing Rigidbody!");
    }

    void FixedUpdate()
    {
        ApplyMotorPhysics();
        ApplySteeringPhysics();
        MoveRobot();
    }

    void ApplyMotorPhysics()
    {
        Debug.Log("physics: " + speedCommand);

        float motorInput = Mathf.Clamp(speedCommand / maxSpeed, -1f, 1f);
        float rpm = motorInput * maxRPM;
        float angularVelocity = (rpm / 60f) * 2f * Mathf.PI;
        float linearSpeed = angularVelocity * wheelRadius;

        currentSpeed = Mathf.Lerp(currentSpeed, linearSpeed, motorAcceleration * Time.fixedDeltaTime);
    }

    void ApplySteeringPhysics()
    {
        float steeringInput = Mathf.Clamp(steeringCommand / maxSteeringCommand, -1f, 1f);
        float targetAngle = steeringInput * maxSteeringAngle;

        currentSteeringAngle = Mathf.Lerp(currentSteeringAngle, targetAngle, steeringSpeed * Time.fixedDeltaTime);
    }

    void MoveRobot()
    {
        // 1. ROTIRE: aplicam rotatie pe baza vitezei unghiulare (grade/secunda)
        float rotationAmount = currentSteeringAngle * Time.fixedDeltaTime;
        Quaternion deltaRotation = Quaternion.Euler(0f, rotationAmount, 0f);
        _rBody.MoveRotation(_rBody.rotation * deltaRotation);

        // 2. DEPLASARE: deplasam inainte pe axa 'forward' a corpului rotit
        Vector3 direction = _rBody.rotation * Vector3.forward;
        _rBody.MovePosition(_rBody.position + direction * currentSpeed * Time.fixedDeltaTime);
    }
}
