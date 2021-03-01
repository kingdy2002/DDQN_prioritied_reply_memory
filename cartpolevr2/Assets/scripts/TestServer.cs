using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Net.Sockets;
using System.Net;
using System.Threading;
using System.Runtime.InteropServices;
using System.IO;
using System.Text;

public struct SendPacket
{
    public float data1; //x위치
    public float data2; //x속도


    public float data3; //z각도
    public float data4; //z각속도

    public bool isDone; //종료확인
    public float hight;
}

public struct RecPacket
{
    public float data; //x 방향힘
}


public class TestServer : MonoBehaviour
{
    // Start is called before the first frame update
    public static TestServer instance;
    TcpListener Server = null;
    TcpClient Client = null;
    NetworkStream Stream = null;
    private int whattime = 0;
    //bool Socket_Threading_Flag = false;

    //SendPacket SendData;
    RecPacket RecData;
    bool isReced;
    Thread Socket_Thread;

    public Testcontroller controller;

    private void Awake()
    {
        instance = this;
        Socket_Thread = new Thread(docker);
        //Socket_Threading_Flag = true;
        Socket_Thread.Start();

        isReced = false;
        RecData = new RecPacket();


    }

    private void FixedUpdate()
    {
        if (isReced)
        {
            Debug.Log("data 받음");
            StartCoroutine(controller.Action(RecData.data));
            isReced = false;
        }
    }
    public static T ByteToStruct<T>(byte[] buffer) where T : struct
    {
        int size = Marshal.SizeOf(typeof(T));
        //Debug.Log(System.String.Format("size is {0} buffer.Length is {1}", size, buffer.Length));
        if (size > buffer.Length)
        {
            throw new Exception();
        }

        IntPtr ptr = Marshal.AllocHGlobal(size);
        Marshal.Copy(buffer, 0, ptr, size);
        T obj = (T)Marshal.PtrToStructure(ptr, typeof(T));
        Marshal.FreeHGlobal(ptr);
        return obj;
    }

    public static byte[] StructToByte(object obj)
    {
        int datasize = Marshal.SizeOf(obj);
        IntPtr buff = Marshal.AllocHGlobal(datasize);
        Marshal.StructureToPtr(obj, buff, false);
        byte[] data = new byte[datasize];
        Marshal.Copy(buff, data, 0, datasize);
        Marshal.FreeHGlobal(buff);
        return data;
    }
    private void docker()
    {
        Int32 Port = 1111;
        IPAddress Addr = IPAddress.Any;
        Server = new TcpListener(Addr, Port);
        Server.Start();
        Debug.Log("소켓 대기중....");
        Client = Server.AcceptTcpClient();
        Debug.Log("소켓 연결되었습니다.");
        Stream = Client.GetStream();
        byte[] Buffer = new byte[1024];

        int length = 0;
        while (true)
        {
            //Debug.Log(".............................");
            try
            {
                //Debug.Log("데이터 받는중 ");
                length =  Stream.Read(Buffer, 0, 1023);
                //Debug.Log(length);

                RecData = ByteToStruct<RecPacket>(Buffer);
            
                isReced = true;

            }
            catch (Exception e)
            {
                Debug.Log(e.Message);
                //ocket_Threading_Flag = false;
                Client.Close();
                Server.Stop();
                continue;
            }
        }
    }
    public void SendMessage(SendPacket sendData)
    {

        //Debug.Log(Marshal.SizeOf(sendData) + " : " + sendData.data1 + " " + sendData.data2 + " " + sendData.data3 + " " + sendData.data4 + " " + sendData.data5 + " " + sendData.data6 + " " + sendData.data7);
        byte[] packetArray = StructToByte(sendData);
        //if(sendStm != null)
        //Debug.Log(whattime);
        whattime++;
        Stream.Write(packetArray, 0, packetArray.Length);
    }
}
