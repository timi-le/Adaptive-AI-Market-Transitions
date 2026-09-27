"""Run each notebook in a fresh kernel; fail on errors and retain executed output."""
from pathlib import Path
import sys,argparse,time,json,hashlib,platform,tempfile
import nbformat
from nbclient import NotebookClient
from traitlets.config import Config

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--kernel',default='python3')
    parser.add_argument('--only',nargs='*',help='Notebook filename prefixes, e.g. 05 06')
    parser.add_argument('--timeout',type=int,default=1800)
    parser.add_argument('--transport',choices=['tcp','ipc'],default='tcp',help='Use ipc on Unix environments without TCP loopback')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    output=root/'executed_notebooks';output.mkdir(exist_ok=True)
    rows=[];start=time.time()
    for path in sorted((root/'notebooks').glob('*.ipynb')):
        if args.only and not any(path.name.startswith(p) for p in args.only):continue
        nb=nbformat.read(path,as_version=4);t=time.time();print('RUN',path.name,flush=True)
        row={'notebook':path.name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        try:
            with tempfile.TemporaryDirectory(prefix='atrx-kernel-') as kernel_dir:
                config=Config()
                if args.transport=='ipc':
                    config.KernelManager.transport='ipc'
                    config.KernelManager.ip=str(Path(kernel_dir)/'socket')
                NotebookClient(nb,config=config,timeout=args.timeout,kernel_name=args.kernel,resources={'metadata':{'path':str(root)}},allow_errors=False).execute()
            cells=[c for c in nb.cells if c.cell_type=='code']
            row.update(status='passed',code_cells=len(cells),executed_code_cells=sum(c.execution_count is not None for c in cells),error_outputs=sum(o.output_type=='error' for c in cells for o in c.outputs))
        except Exception as e:
            row.update(status='failed',error=str(e));raise
        finally:
            nbformat.write(nb,output/path.name);row['elapsed_seconds']=round(time.time()-t,3);rows.append(row)
            report={'method':'nbclient, separate fresh Jupyter kernel per notebook','python':platform.python_version(),'kernel':args.kernel,'transport':args.transport,'partial_selection':bool(args.only),'notebooks':rows,'all_selected_passed':all(x['status']=='passed' for x in rows),'elapsed_seconds':round(time.time()-start,3)}
            (root/'results/notebook_execution.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PASS',path.name,row['elapsed_seconds'],'seconds',flush=True)
    print('Completed',len(rows),'notebooks',flush=True)

if __name__=='__main__':main()
