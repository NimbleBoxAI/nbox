from logging import getLogger
logger = getLogger()

TIMEOUT_CALLS = 60


class Subway():
    """Simple code that allows extending things by .attr.ing them"""
    def __init__(self, _url, _session):
        self._url = _url.rstrip('/')
        self._session = _session

    def __repr__(self):
        return f"<Subway ({self._url})>"

    def __getattr__(self, attr):
        return Subway(f"{self._url}/{attr}", self._session)

    def __call__(self, method = "get", trailing = "", data = None, _verbose = False):
        fn = getattr(self._session, method)
        url = f"{self._url}{trailing}"
        if _verbose:
            print(f"Calling {url}")
        r = fn(url, json = data)
        if _verbose:
            print(r.content.decode())
        r.raise_for_status() # good when server is good
        return r.json()


class SpecSubway():
    """Subway is simple when starting, but what if we had a spec that could perform sanity checks
    and parse data when __call__ ed. In this version it can read an OpenAPI spec and perform lazy
    checking"""
    def __init__(self, _url, _session, _spec, __name = None):
        self._url = _url.rstrip('/')
        self._session = _session
        self._spec = _spec
        self._name = __name
        self._caller = (
            (len(_spec) == 3 and set(_spec) == set(["method", "meta", "src"])) or
            (len(_spec) == 4 and set(_spec) == set(["method", "meta", "src", "response_kwargs_dict"])) or
            "/" in self._spec
        )

    @classmethod
    def from_openapi(cls, openapi, _url, _session):
        paths = openapi["paths"]
        spec = openapi["components"]["schemas"]
        
        tree = {}
        for p in tuple(paths.keys()):
            t = tree
            for part in p.split('/')[1:]:
                part = "/" if part == "" else part
                t = t.setdefault(part, {})

        def _dfs(tree, trail = []):
            for t in tree:
                if tree[t] == {}:
                    src = "/" + "/".join(trail)
                    if t!= "/":
                        src = src + "/" if src != "/" else src
                        src = src + t
                    
                    try:
                        data = paths[src]
                    except:
                        src = src + "/"
                        data = paths[src]
                    method = tuple(data.keys())[0]
                    body = data[method]
                    dict_ = {"method": method, "meta": None, "src": src}
                    if "requestBody" in body:
                        schema_ref = body["requestBody"]["content"]["application/json"]["schema"]["$ref"].split("/")[-1]
                        _req_body = spec[schema_ref]
                        kwargs_dict = list(_req_body["properties"])
                        dict_["meta"] = {
                            "kwargs_dict": kwargs_dict,
                            "required": _req_body.get("required", None)
                        }
                    if "responses" in body:
                        schema = body["responses"]["200"]["content"]["application/json"]["schema"]
                        if "$ref" in schema:
                            schema_ref = schema["$ref"].split("/")[-1]
                            _req_body = spec[schema_ref]
                            kwargs_dict = list(_req_body["properties"])
                            if dict_["meta"] != None:
                                dict_["meta"].update({"response_kwargs_dict": kwargs_dict})
                            else:
                                dict_["meta"] = {"response_kwargs_dict": kwargs_dict}
                    tree[t] = dict_
                else:
                    _dfs(tree[t], trail + [t])

        _dfs(tree)
        
        return cls(_url, _session, tree)

    def __repr__(self):
        return f"<SpecSubway ({self._url})>"

    def __getattr__(self, attr):
        # https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute
        if self._caller and len(self._spec) == 1:
            raise AttributeError(f"'.{self._name}' does not have children")
        if attr not in self._spec:
            raise AttributeError(f"'.{attr}' is not a valid function")
        return SpecSubway(f"{self._url}/{attr}", self._session, self._spec[attr], attr)
    
    def __call__(self, *args, _verbose = False, _parse = False, **kwargs):
        if not self._caller:
            raise AttributeError(f"'.{self._name}' is not an endpoint")
        spec = self._spec
        if self._caller and "/" in self._spec:
            spec = self._spec["/"]
        
        data = None
        if spec["meta"] == None:
            assert len(args) == len(kwargs) == 0, "This method does not accept any arguments"
        else:
            spec_meta = spec["meta"]
            if "kwargs_dict" not in spec_meta:
                assert len(args) == len(kwargs) == 0, "This method does not accept any arguments"
            else:
                kwargs_dict = spec["meta"]["kwargs_dict"]
                required = spec["meta"]["required"]
                data = {}
                for i in range(len(args)):
                    if required != None:
                        data[required[i]] = args[i]
                    else:
                        data[kwargs_dict[i]] = args[i]
                for key in kwargs:
                    if key not in kwargs_dict:
                        raise AttributeError(f"{key} is not a valid argument")
                    data[key] = kwargs[key]
                if required != None:
                    for key in required:
                        if key not in data:
                            raise AttributeError(f"{key} is a required argument")

        fn = getattr(self._session, spec["method"])
        url = f"{self._url}"
        if self._caller and "/" in self._spec:
            url += "/"
        if _verbose:
            print(f"{spec['method'].upper()} {url}")
            print("-->>", data)
        r = fn(url, json = data)
        if not r.status_code == 200:
            raise ValueError(r.content.decode())
        
        out = r.json()
        if _parse and self._spec["meta"] != None and "response_kwargs_dict" in self._spec["meta"]:
            out = [out[k] for k in self._spec["meta"]["response_kwargs_dict"]]
            if len(out) == 1:
                return out[0]
        return out

