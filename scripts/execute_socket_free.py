"""Execute notebooks through IPykernel's in-process Jupyter channels.

Each notebook runs in its own Python process, so no notebook shares hidden
interpreter state. This mode uses no network/IPC sockets. The normal nbclient
runner remains available as execute_notebooks.py.
"""
from pathlib import Path
import argparse,json,os,sys,time,subprocess,hashlib,platform
from queue import Empty
import nbformat

def execute_one(path):
    from ipykernel.inprocess.manager import InProcessKernelManager
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    output=root/'executed_notebooks';output.mkdir(exist_ok=True)
    km=InProcessKernelManager();km.start_kernel();client=km.client();client.start_channels()
    failed=False
    def run(source):
        msgid=client.execute(source,store_history=True,allow_stdin=False,stop_on_error=True)
        reply=client.get_shell_msg(timeout=1800)
        if reply['parent_header'].get('msg_id')!=msgid:raise RuntimeError('Unexpected shell reply')
        messages=[]
        while True:
            message=client.get_iopub_msg(timeout=1800)
            if message['parent_header'].get('msg_id')!=msgid:continue
            if message['msg_type']=='status' and message['content']['execution_state']=='idle':break
            if message['msg_type'] in ['stream','display_data','execute_result','error']:
                messages.append(nbformat.v4.output_from_msg(message))
        return reply['content'],messages
    try:
        init,_=run('%matplotlib inline')
        if init['status']!='ok':raise RuntimeError('Unable to initialize inline backend')
        for index,cell in enumerate(nb.cells):
            if cell.cell_type!='code':continue
            print(f'  cell {index}',flush=True)
            reply,cell.outputs=run(cell.source);cell.execution_count=reply['execution_count']
            if reply['status']!='ok':
                failed=True;print('CELL ERROR',reply,flush=True);break
    finally:
        nbformat.validate(nb);nbformat.write(nb,output/path.name)
        client.stop_channels();km.shutdown_kernel()
    return not failed

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--single',type=Path);parser.add_argument('--only',nargs='*');args=parser.parse_args()
    if args.single:raise SystemExit(0 if execute_one(args.single.resolve()) else 1)
    root=Path(__file__).resolve().parents[1];rows=[];started=time.time()
    for path in sorted((root/'notebooks').glob('*.ipynb')):
        if args.only and not any(path.name.startswith(p) for p in args.only):continue
        print('RUN',path.name,flush=True);t=time.time()
        env={**os.environ,'PYTHONHASHSEED':'0','PYDEVD_DISABLE_FILE_VALIDATION':'1'}
        p=subprocess.run([sys.executable,'-Xfrozen_modules=off',str(Path(__file__).resolve()),'--single',str(path)],cwd=root,env=env,timeout=3600)
        nb=nbformat.read(root/'executed_notebooks'/path.name,as_version=4);cells=[c for c in nb.cells if c.cell_type=='code']
        row={'notebook':path.name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':'passed' if p.returncode==0 else 'failed','code_cells':len(cells),'executed_code_cells':sum(c.execution_count is not None for c in cells),'error_outputs':sum(o.output_type=='error' for c in cells for o in c.outputs),'elapsed_seconds':round(time.time()-t,3)}
        rows.append(row)
        report={'method':'IPykernel in-process Jupyter channels; one isolated Python process per notebook; socket-free execution','python':platform.python_version(),'normal_socket_kernel_status':'Unavailable here: TCP and IPC bind operations are prohibited. Use execute_notebooks.py on your own computer.','partial_selection':bool(args.only),'notebooks':rows,'all_selected_passed':all(r['status']=='passed' for r in rows),'elapsed_seconds':round(time.time()-started,3)}
        (root/'results/notebook_execution.json').write_text(json.dumps(report,indent=2)+'\n')
        print(row['status'].upper(),path.name,row['elapsed_seconds'],'seconds',flush=True)
        if p.returncode:raise SystemExit(p.returncode)
    print('Completed',len(rows),'notebooks',flush=True)

if __name__=='__main__':main()
