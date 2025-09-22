import os
import re
import json
from cfile_parse import CParser
from node_prompt import CProjectSearcher
from utils import DS_REPO_DIR, DS_FILE, DS_GRAPH_DIR


class CProjectParser(object):
    def __init__(self):
        self.c_parser = CParser()
        self.file_pattern = re.compile(r'[^\w\-]')
        self.header_pattern = re.compile(r'\.(h|hpp)$')
        self.source_pattern = re.compile(r'\.(c|cpp)$')
        
        self.proj_searcher = CProjectSearcher()
        
        self.proj_dir = None
        self.parse_res = None
    
    def set_proj_dir(self, dir_path):
        if not dir_path.endswith(os.sep):
            self.proj_dir = dir_path + os.sep
        else:
            self.proj_dir = dir_path
    
    def retain_project_rels(self):
        for module, file_info in self.parse_res.items():
            for name, info_dict in file_info.items():
                struct_name = info_dict.get("in_struct", None)
                
                rels = info_dict.get("rels", None)
                if rels is not None:
                    del_index = []
                    for i, item in enumerate(rels):
                        if len(item) == 2:
                            find_info = self.proj_searcher.name_in_file(item[0], list(file_info), name, struct_name)
                            if find_info is None:
                                del_index.append(i)
                            else:
                                info_dict["rels"][i] = [find_info[0], find_info[1], item[1]]
                        else:
                            find_info = self.proj_searcher.name_in_file(item[0], list(file_info), name, struct_name)
                            if find_info is None:
                                del_index.append(i)
                            else:
                                pass
                    
                    for index in reversed(del_index):
                        info_dict["rels"].pop(index)
                    
                    if len(info_dict["rels"]) == 0:
                        info_dict.pop("rels")

                include_info = info_dict.get("include", None)
                if info_dict["type"] == 'Variable' and include_info is not None:
                    judge_res = self.proj_searcher.is_local_include(module, include_info)
                    if judge_res is None:
                        info_dict.pop("include")
                    else:
                        info_dict["include"] = judge_res
    
    def _get_all_c_file_paths(self, target_path):
        if not os.path.isdir(target_path):
            return {}
        
        dir_list = [target_path,]
        c_dict = {} 
        
        while len(dir_list) > 0:
            c_dir = dir_list.pop()
            c_dict[c_dir] = set()
            
            for item in os.listdir(c_dir):
                if item.startswith('.'): 
                    continue
                    
                fpath = os.path.join(c_dir, item)
                if os.path.isdir(fpath):
                    if re.search(self.file_pattern, item) is None:
                        dir_list.append(fpath)
                        c_dict[c_dir].add(fpath)
                elif os.path.isfile(fpath) and (self.header_pattern.search(fpath) or self.source_pattern.search(fpath)):
                    if re.search(self.file_pattern, os.path.splitext(item)[0]) is None:
                        c_dict[c_dir].add(fpath)
        
        return c_dict
    
    def _get_module_name(self, fpath):
        rel_path = fpath[len(self.proj_dir):]
        return rel_path
    
    def parse_dir(self, c_proj_dir):
        self.set_proj_dir(c_proj_dir)
        c_dict = self._get_all_c_file_paths(c_proj_dir)
        
        c_files = set()
        for dir_path, file_set in c_dict.items():
            for fpath in file_set:
                if os.path.isfile(fpath) and (self.header_pattern.search(fpath) or self.source_pattern.search(fpath)):
                    c_files.add(fpath)
        
        self.parse_res = {}
        for fpath in c_files:
            module = self._get_module_name(fpath)
            
            try:
                self.c_parser.set_file_path(fpath)
                info_dict = self.c_parser.parse(fpath)
                
                if info_dict and len(info_dict) > 0:
                    self.parse_res[module] = info_dict
            except Exception as e:
                print(f"Error parsing {fpath}: {e}")
        
        self.proj_searcher.set_proj(c_proj_dir, self.parse_res)
        self.retain_project_rels()
        
        return self.parse_res


if __name__ == '__main__':
    
    with open(DS_FILE, 'r') as f:
        ds = [json.loads(line) for line in f.readlines()]
    
    pkg_set = set([x['pkg'] for x in ds])
    print(f'There are {len(pkg_set)} repositories in dataset.')
    
    project_parser = CProjectParser()
    
    if not os.path.isdir(DS_GRAPH_DIR):
        os.mkdir(DS_GRAPH_DIR)
    
    for item in os.listdir(DS_REPO_DIR):
        if item not in pkg_set:
            continue
        
        dir_path = os.path.join(DS_REPO_DIR, item)
        if os.path.isdir(dir_path):
            try:
                info = project_parser.parse_dir(dir_path)
                
                with open(os.path.join(DS_GRAPH_DIR, f'{item}.json'), 'w') as f:
                    json.dump(info, f)
            except Exception as e:
                print(f"Error processing {item}: {e}")
    
    import os

    visible_files = [
        f for f in os.listdir(DS_GRAPH_DIR)
        if not f.startswith('.') 
        and os.path.isfile(os.path.join(DS_GRAPH_DIR, f))  
    ]

    print(f'Generated repo-specific context graph for {len(visible_files)} repositories.')

    