using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Management;

namespace ConsoleApplication1
{
    /*
[event_receiver(managed)]   // optional for native C++ and managed classes
__gc class CReceiver {
    public:
   void BlinkHandler(int nValue) {
   }
   void hookEvent(CSource* pSource) {
      __hook(&CSource::BlinkEvent, pSource, &CReceiver::BlinkHandler);
   }
   void unhookEvent(CSource* pSource) {
      __unhook(&CSource::BlinkEvent, pSource, &CReceiver::BlinkHandler);
   }
}*/

    unsafe public class webCamWorker
    {
        [DllImport("goErgo.dll")]
        static extern int webCamMain();
        [DllImport("goErgo.dll")]
        static extern int get_stats(int* blink, int* ambient_alarm, int* posture_alarm, int use_buf);
        public void DoWork()
        {
            webCamMain();
        }

        public void GetStats()
        {
            while (true)
            {

            Thread.Sleep(1000);
            int blink, ambient_alarm, posture_alarm;
            get_stats(&blink, &ambient_alarm, &posture_alarm, 1);
            Console.Out.WriteLine("Blink:" + blink + " Ambient light alarm:" + ambient_alarm + " posture_alarm:" + posture_alarm);
            }
        }
    }
    class Program
    {
        [DllImport("dxva2.dll", EntryPoint = "GetMonitorBrightness", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool GetMonitorBrightness(
            [In]IntPtr hMonitor, ref uint pdwMinimumBrightness, ref uint pdwCurrentBrightness, ref uint pdwMaximumBrightness);

        /*
        static int getBrightness()
        {
            uint pdwMinimumBrightness = NULL;
            uint pdwMaximumBrightness = NULL;
            IntPtr pmh = pPhysicalMonitors[0].hPhysicalMonitor;
            GetMonitorBrightness(pmh, pdwMinimumBrightness, pdwMaximumBrightness);

            DWORD dw;
            HMONITOR hMonitor = NULL;
            DWORD cPhysicalMonitors;
            LPPHYSICAL_MONITOR pPhysicalMonitors = NULL;

            LPDWORD pdwMinimumBrightness = NULL;
            LPDWORD pdwCurrentBrightness = NULL;
            LPDWORD pdwMaximumBrightness = NULL;

            HWND hwnd = FindWindow(NULL, NULL);

            hMonitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONULL);

            pPhysicalMonitors = (LPPHYSICAL_MONITOR)malloc(cPhysicalMonitors * sizeof(PHYSICAL_MONITOR));
            HANDLE pmh = pPhysicalMonitors[0].hPhysicalMonitor;
            bSuccess = GetMonitorBrightness(hMonitor, pdwMinimumBrightness, pdwCurrentBrightness, pdwMaximumBrightness);
        }
             * */
        static void Main(string[] args)
        {
            webCamWorker worker = new webCamWorker();
            Thread camThread = new Thread(new ThreadStart(worker.DoWork));
            camThread.Start();
            webCamWorker statsWorker = new webCamWorker();
            Thread getStatsThread = new Thread(new ThreadStart(statsWorker.GetStats));
            getStatsThread.Start();
            getStatsThread.IsBackground = true;
        }
    }
}
