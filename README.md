# Hate Speech Detection in Turkish News using Target-Oriented Linguistic Features

Implementation of hate speech detection task for Turkish news in print media. Rulemodel should be downloaded and be put in a folder whose name is rulemodel.

## Endpoint with fastAPI
If you start the app with following command, you can run the fastAPI app to use `detect` endpoint. You must send body of the news in `text` parameter.
```bash
$ uvicorn app:app --reload --host <host_ip> --port <port_number>
```

## Demo with Streamlit
You can use the demo by running the following command.
```bash
$ streamlit run  demo.py --server.port <port_number>
```