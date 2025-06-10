using UnityEngine;

public class RobotController : MonoBehaviour
{
    public RobotPhysicsController robotPhysics;
    public float pixelToUnityFactor = 0.01f;

    [Header("PID Movement")]
    public int desired_distance_movement = 40; // ținta de bază în pixeli
    public float kp_movement = 0.6f;
    public float ki_movement = 0f;
    public float kd_movement = 0.01f;
    float previous_error_movement = 0f;
    float integral_movement = 0f;

    [Header("Movement Margin (pixeli)")]
    public int minMovementDistance = 40;
    public int maxMovementDistance = 50;

    [Header("PID Rotation")]
    public int desired_distance_rotation = 511; // ținta de bază în pixeli
    public float kp_rotation = 0.6f;
    public float ki_rotation = 0f;
    public float kd_rotation = 0.01f;
    float previous_error_rotation = 0f;
    float integral_rotation = 0f;

    [Header("Rotation Margin (pixeli)")]
    public int minRotationDistance = 501;
    public int maxRotationDistance = 521;

    void Start()
    {
        if (robotPhysics == null)
        {
            robotPhysics = GetComponent<RobotPhysicsController>();
            if (robotPhysics == null)
                Debug.LogError("RobotPhysicsController component not found!");
        }
    }

    public void ProcessCommand(string command)
    {
        Debug.Log("Processing command: " + command);

        string[] parts = command.Split('|');
        if (parts.Length != 2) return;

        string rotationCommand = parts[0];
        string movementCommand = parts[1];

        ProcessMovementCommand(movementCommand);
        ProcessRotationCommand(rotationCommand);
    }

    void ProcessMovementCommand(string command)
    {
        if (command.ToLower() == "none")
        {
            robotPhysics.speedCommand = 0;
            return;
        }

        string[] parts = command.Split('#');
        if (parts.Length != 2) return;

        int distance = int.Parse(parts[1]);

        // Dacă suntem în intervalul dorit, oprim mișcarea
        if (distance >= minMovementDistance && distance <= maxMovementDistance)
        {
            robotPhysics.speedCommand = 0;
            return;
        }

        float error = (desired_distance_movement - distance) * pixelToUnityFactor;

        //integral_movement += error * Time.fixedDeltaTime;
        //float derivative = (error - previous_error_movement) / Time.fixedDeltaTime;
        //previous_error_movement = error;

        //float speedOutput = kp_movement * error + ki_movement * integral_movement + kd_movement * derivative;
        //robotPhysics.speedCommand = speedOutput;

        float absErr = Mathf.Abs(error);
        float kp = (absErr > 20f) ? kp_movement : kp_movement * 0.4f;
        float kd = kd_movement * 1.2f;
        float maxLimit = (absErr > 20f) ? 1f : 0.15f;

        integral_movement += error * Time.fixedDeltaTime;
        float derivative = (error - previous_error_movement) / Time.fixedDeltaTime;
        previous_error_movement = error;

        float rawOutput = kp * error + ki_movement * integral_movement + kd * derivative;
        float clamped = Mathf.Clamp(rawOutput, -maxLimit, maxLimit);

        // RATE LIMIT
        robotPhysics.speedCommand = Mathf.MoveTowards(
            robotPhysics.speedCommand,
            clamped,
            0.3f * Time.fixedDeltaTime
        );
    }

    void ProcessRotationCommand(string command)
    {
        if (command.ToLower() == "none")
        {
            robotPhysics.steeringCommand = 0;
            return;
        }

        string[] parts = command.Split('#');
        if (parts.Length != 2) return;

        int distance = int.Parse(parts[1]);

        // Dacă suntem în intervalul dorit, oprim rotația
        if (distance >= minRotationDistance && distance <= maxRotationDistance)
        {
            robotPhysics.steeringCommand = 0;
            return;
        }

        float error = (desired_distance_rotation - distance) * pixelToUnityFactor;

        integral_rotation += error * Time.fixedDeltaTime;
        float derivative = (error - previous_error_rotation) / Time.fixedDeltaTime;
        previous_error_rotation = error;

        float angleOutput = kp_rotation * error + ki_rotation * integral_rotation + kd_rotation * derivative;
        robotPhysics.steeringCommand = angleOutput;
    }
}
