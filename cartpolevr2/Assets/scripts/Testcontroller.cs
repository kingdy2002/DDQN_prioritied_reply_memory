using System.Collections;
using System;
using System.Collections.Generic;
using UnityEngine.SceneManagement;
using UnityEngine;

public class State
{
    public float x_position = 0;
    public float x_speed = 0;

    public float z_angle = 0;
    public float last_z_angle = 0;
    public float z_angle_speed = 0;


    public float hight = 6.5f;

    public bool isDone = false;
    public int episode = 0;


    public void reset()
    {
        x_position = 0;
        x_speed = 0;





        z_angle = 0;
        last_z_angle = 0;
        z_angle_speed = 0;


        hight = 6.5f;
        isDone = false;
        episode = 0;
    }
    public void CalculateAngularAcc()
    {
        z_angle_speed = (z_angle - last_z_angle) / Time.deltaTime;
        last_z_angle = z_angle;
    }

}
public class Testcontroller : MonoBehaviour
{
    // Start is called before the first frame update
    public float failAngel = 70;
    public int max_Step = 100;
    public Rigidbody pole;
    public Rigidbody ball;
    public Rigidbody cart;

    private SendPacket SendPacket;
    private State state;
    private CharacterController controller;

    private Vector3 cart_orig;
    private Vector3 pole_orig;
    private Vector3 ball_orig;

    public Transform pivotTransfrom;

    private int epoch = 0;


    void Start()
    {
        controller = GetComponent<CharacterController>();
        cart = GetComponent<Rigidbody>();
        SendPacket = new SendPacket();
        state = new State();


        cart_orig = cart.transform.position;
        pole_orig = pole.transform.position;
        ball_orig = ball.transform.position;

    }
    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Return))
            StartCoroutine(StartCartpole());
    }

    IEnumerator StartCartpole()
    {
        GiveRandomForce();
        UpdateState();
        Debug.Log("StartCartpole");
        TestServer.instance.SendMessage(ConvertData());
        yield return new WaitForFixedUpdate();
    }
    IEnumerator StartResetEpisode()
    {
        yield return new WaitForFixedUpdate();
        ResetEpisode();

    }
    public IEnumerator Action(float action)
    {
        
        Vector3 vec;

        if (action <= 0.5f)
        {
            vec = Vector3.right;
        }
        else
        {
            vec = -Vector3.right;
        }

        
        //Debug.Log("action :" + action + " episode : " + state.episode + " epoch : " + epoch + " " + vec);

        cart.AddForce(vec* 100);
        state.episode++;
        yield return new WaitForFixedUpdate();
        UpdateState();
        CheckFail();





    }

    void GiveRandomForce()
    {
        System.Random r = new System.Random();
        Vector3 Force = new Vector3(r.Next(-50, 50), 0, 0);
        cart.AddForce(Force);
    }
    
   

    bool IsMaxStep()
    {
        if (state.episode >= max_Step)
            return true;
        return false;
    }
    bool IsOnPlane()
    {
        Vector3 abs_pos = cart.transform.position - pivotTransfrom.position;
        float abs_x_pos = Math.Abs(abs_pos.x);
        if (abs_x_pos > 12)
            return false;
        return true;
    }
    public void CheckFail()
    {

        if (!IsOnPlane() || (ball.transform.position.y - cart.transform.position.y) < 4)
        {
            state.isDone = true;
            //한 에피소드씩 재생할 때 사용
            //StartCoroutine(StartCartpole());
            Debug.Log(String.Format("에피소드 종료 최대 episode는 {0} 현재 epoch는 {1}", state.episode, epoch));
            TestServer.instance.SendMessage(ConvertData());
            StartCoroutine(StartResetEpisode());
            epoch++;
        }
        else if(state.episode >= 500) {
            state.isDone = true;
            //한 에피소드씩 재생할 때 사용
            //StartCoroutine(StartCartpole());
            Debug.Log(String.Format("에피소드 종료 최대 episode는 {0} 현재 epoch는 {1}", state.episode, epoch));
            TestServer.instance.SendMessage(ConvertData());
            StartCoroutine(StartResetEpisode());
            epoch++;
        }
        else
        {
            Debug.Log("CheckFail");
            TestServer.instance.SendMessage(ConvertData());
        }


    }
    void UpdateState()
    {
        state.x_position = pivotTransfrom.position.x -  cart.position.x;
        state.x_speed = cart.velocity.x;
        float hei = ball.position.y - cart.position.y;
        float underx = ball.position.x - cart.position.x;
        state.z_angle = Mathf.Atan2(underx, hei) * Mathf.Rad2Deg;
        state.CalculateAngularAcc();


    }
    // Update is called once per frame
    void ResetEpisode()
    {
        state.reset();
        cart.transform.position = cart_orig;
        pole.transform.position = pole_orig;
        ball.transform.position = ball_orig;

        pole.transform.rotation = Quaternion.identity;
        ball.transform.rotation = Quaternion.identity;
        cart.transform.rotation = Quaternion.identity;

        pole.velocity = Vector3.zero;
        ball.velocity = Vector3.zero;
        cart.velocity = Vector3.zero;

        pole.angularVelocity = Vector3.zero;
        ball.angularVelocity = Vector3.zero;
        cart.angularVelocity = Vector3.zero;
        state.isDone = false;

        GiveRandomForce();
        UpdateState();
        //Debug.Log("ResetEpisode");
        TestServer.instance.SendMessage(ConvertData());
    }
    public SendPacket ConvertData()
    {
        SendPacket.data1 = state.x_position;
        SendPacket.data2 = state.x_speed * 0.1f;


        SendPacket.data3 = Mathf.Deg2Rad * state.z_angle;
        SendPacket.data4 = state.z_angle_speed * 0.1f;

        SendPacket.isDone = state.isDone;
        SendPacket.hight = state.hight;

        return SendPacket;
    }
}