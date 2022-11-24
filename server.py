import socket
import search
import util
import time
#Here we use localhost ip address
# and port number
LOCALHOST = "127.0.0.1"
PORT = 8080
# calling server socket method
server = socket.socket(socket.AF_INET,
                       socket.SOCK_STREAM)
server.bind((LOCALHOST, PORT))
server.listen(1)

postion_lookup_table_file_name = "./data/postion_lookup_table.p"
lookup_table =  util.read_data(postion_lookup_table_file_name)
with open("./data/index_table.txt","rb") as f:
    search_engine = search.Search_engine(f,lookup_table=lookup_table)
    print("Server started")
    print("Waiting for client request..")
    # Here server socket is ready for
    # get input from the user
    clientConnection, clientAddress = server.accept()
    print("Connected client :", clientAddress)
    msg = ''
    # Running infinite loop
    while True:
        data = clientConnection.recv(1024)
        msg = data.decode()
        if msg == '':
            print("Connection is Over")
            break
    
        print(f"Query {msg}")
        begin = time.time()
        result = search_engine.search(msg)
        end = time.time()
        print(f"Query time {end-begin:.3f}")
        # Here we change int to string and
        # after encode send the output to client
        output = str(result)
        clientConnection.send(output.encode())
    clientConnection.close()



