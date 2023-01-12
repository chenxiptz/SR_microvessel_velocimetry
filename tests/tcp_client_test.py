import threading
import time
import socket
import sys

remaining_frames = 0

def listen(sock):
    global remaining_frames
    data = int.from_bytes(sock.recv(4), 'big', signed=True)
    remaining_frames += 1
    print(f'listen, {remaining_frames}')
    return data

def process():
    global remaining_frames
    remaining_frames -= 1
    time.sleep(0.2)



def thread_task_listen(lock, sock):
    """
    task for thread
    calls increment function 100000 times.
    """
    data = 0
    while data != -1:
        lock.acquire()
        data = listen(sock)
        lock.release()

def thread_task_process(lock):
    """
    task for thread
    calls increment function 100000 times.
    """
    global remaining_frames
    while remaining_frames >= 0:
        lock.acquire()
        data = process()
        lock.release()
        print(f'process, {remaining_frames}')
  
def main_task():
    global remaining_frames
    # setting global variable x as 0
    remaining_frames = 0
  
    # creating a lock
    lock = threading.Lock()
  
    # creating threads
    t1 = threading.Thread(target=thread_task_listen, args=(lock, sock,))
    t2 = threading.Thread(target=thread_task_process, args=(lock,))
  
    # start threads
    t1.start()
    t2.start()
  
    # wait until threads finish their job
    t1.join()
    t2.join()
  
if __name__ == "__main__":
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ('localhost', 8192)
    print(sys.stderr)
    sock.connect(server_address)
    for i in range(1):
        main_task()
        print("Iteration {0}: x = {1}".format(i,remaining_frames))