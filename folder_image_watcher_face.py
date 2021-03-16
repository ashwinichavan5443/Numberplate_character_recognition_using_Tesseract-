import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class Watcher:
    DIRECTORY_TO_WATCH = "/home/cloudvms/out-test/"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(0.0001)
        except:
            self.observer.stop()
            print (Error)

        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        #elif event.event_type == 'created':
            # Take any action here when a file is first created.
            #print ("Received created event -" , event.src_path)

        elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            #print ("Received modified event - " , event.src_path)
            cmd1=  "ftp-upload -h 45.79.121.55 -u ftpuser --passive --password neon@123 -d abc/ " + event.src_path
            image_name = "abc/"+event.src_path[29:]
            image_name2 = event.src_path[29:]
            print(image_name2)
            cam_did,sep,rem_name=image_name2.partition('_')
            person_c,sep,rem_name =rem_name.partition('_')
            print(image_name2,cam_did,person_c,rem_name)
            print (cmd1)
            cmd="nohup wget \"" +"http://testcloudvms.azurewebsites.net/aiDataAPI.asmx/AddNotificatinData?uuid="+cam_did+"&aid=4"+"&data="+image_name+"&hcount="+person_c+"&datamsg=Head Count="+person_c+"\""
            os.system(cmd1)
            #os.system(cmd)

if __name__ == '__main__':
    w = Watcher()
    w.run()
