using UnityEngine;

public class RobotController : MonoBehaviour
{
    Rigidbody _rBody;

    float _speed = 0f;
    float _angle = 0f;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        if (GetComponent<Rigidbody>())
        {
            _rBody = GetComponent<Rigidbody>();
        } else
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
        if (command.ToLower() == "none")
        {
            _speed = 0f;
            return;
        }

        // command: action#number
        string[] parts = command.Split('#');
        string action = parts[0].ToLower();
        float factor = int.Parse(parts[1]);

        float kp = 0.01f;

        if (action == "forward")
        {
            _speed += factor * kp;
        }
        else if (action == "backward")
        {
            _speed -= factor * kp;
        }
        else
        {
            Debug.LogError("Unknown action: " + action);
            return;
        }
    }

    void ProcessRotationCommand(string command)
    {
        if (command.ToLower() == "none")
        {
            return;
        }

        // command: action#number
        string[] parts = command.Split('#');
        string action = parts[0].ToLower();
        float factor = int.Parse(parts[1]);

        float kp = 0.1f;

        if (action == "left")
        {
            _angle += factor * kp;
        }
        else if (action == "right")
        {
            _angle -= factor * kp;
        }
        else
        {
            Debug.LogError("Unknown action: " + action);
            return;
        }
    }

    void FixedUpdate()
    {
        // Move the object forward along its forward axis

        transform.position += transform.forward * _speed * Time.fixedDeltaTime;
        _rBody.MoveRotation(Quaternion.Euler(0, _angle, 0));
    }
}
