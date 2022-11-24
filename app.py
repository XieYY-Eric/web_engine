from flask import Flask,redirect,url_for,render_template,request,session
import util
import search

app = Flask(__name__)
app.secret_key = "darklord"
search_engine = None

@app.route("/",methods=["GET","POST"])
def home():
    if request.method == "POST":
        query = request.form["query"]
        session["query"] = search_engine.search(query)
        # session["query"] = ["helloWorld","yes","no","222","darkLord"]
        return redirect(url_for("search_result"))
    else:
        return render_template("index.html")


@app.route("/queryResult")
def search_result():
    if "query" in session:
        result = session["query"]
        return render_template("result.html",queryname="result",content = result)
    else:
        return redirect(url_for("home"))


def init_server():
    global search_engine
    postion_lookup_table_file_name = "./data/postion_lookup_table.p"
    lookup_table =  util.read_data(postion_lookup_table_file_name)
    f = open("./data/index_table.txt","rb")
    search_engine = search.Search_engine(f,lookup_table=lookup_table)



####flask web applicaiton
####run this program to launch
if __name__ == "__main__":
    #init server --> get search program start working, gather lookup table and etc
    init_server()
    print("Server is running")
    app.run(debug=True)