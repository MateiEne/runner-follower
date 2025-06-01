using UnityEngine;

public class RobotController : MonoBehaviour
{
    Rigidbody _rBody;

    float _speed = 0f;
    int desired_distance_movement = 40; // in pixels
    float kp_movement = 1.0f;
    float ki_movement = 0;
    float kd_movement = 0.01f;
    float previous_error_movement = 0;
    float integral_movement = 0;


    float _angle = 0f;
    int desired_distance_rotation = 0; // in pixels
    float kp_rotation = 0.6f;
    float ki_rotation = 0;
    float kd_rotation = 0.01f;
    float previous_error_rotation = 0;
    float integral_rotation = 0;

    // Convertor pixeli -> unitati Unity (ajustabil în functie de FOV și rezolutie)
    public float pixelToUnityFactor = 0.01f;


    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        if (GetComponent<Rigidbody>())
        {
            _rBody = GetComponent<Rigidbody>();
        }
        else
        {
            Debug.LogError("No rigidbody!");
        }
    }

    public void ProcessCommand(string command)
    {
        Debug.Log("Processing command: " + command);

        string[] parts = command.Split('|');
        string rotationCommand = parts[0];
        string movementCommand = parts[1];

        ProcessMovementCommand(movementCommand);
        ProcessRotationCommand(rotationCommand);
    }

    void ProcessMovementCommand(string command)
    {
        //  command: distance#number
        if (command.ToLower() == "none")
        {
            return;
        }

        string[] parts = command.Split('#');
        string action = parts[0].ToLower();
        int distance = int.Parse(parts[1]);

        float error = (desired_distance_movement - distance) * pixelToUnityFactor;

        integral_movement += error * Time.fixedDeltaTime;
        float derivative = (error - previous_error_movement) / Time.fixedDeltaTime;

        previous_error_movement = error;

        _speed = kp_movement * error + ki_movement * integral_movement + kd_movement * derivative;
    }

    void ProcessRotationCommand(string command)
    {
        if (command.ToLower() == "none")
        {
            return;
        }

        // command: distance#number
        string[] parts = command.Split('#');
        string action = parts[0].ToLower();
        int distance = int.Parse(parts[1]);

        float error = (desired_distance_rotation - distance) * pixelToUnityFactor;

        integral_rotation += error * Time.fixedDeltaTime;
        float derivative = (error - previous_error_rotation) / Time.fixedDeltaTime;

        previous_error_rotation = error;

        _angle = kp_rotation * error + ki_rotation * integral_rotation + kd_rotation * derivative;

        Debug.Log(_angle);
    }

    void FixedUpdate()
    {
        // Move the object forward along its forward axis
        _rBody.MovePosition(_rBody.position + transform.forward * _speed * Time.fixedDeltaTime);
        Quaternion deltaRotation = Quaternion.Euler(0, _angle, 0);
        _rBody.MoveRotation(_rBody.rotation * deltaRotation);
    }
}
