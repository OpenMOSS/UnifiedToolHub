from datetime import datetime
import inspect
import json
import os
import time
from urllib import request
from urllib.parse import urlencode

WAITING_TIME = 0.5
MAX_OPS = 500  # 由飞书规定的单次最大操作数量


class LarkReport:

    def __init__(self, app_id, app_secret, app_verification_token, bitable_url=None, app_address=""):
        self.storage = {}
        self.APP_ID = app_id
        self.APP_SECRET = app_secret
        self.APP_VERIFICATION_TOKEN = app_verification_token
        self.APP_ADDRESS = app_address
        self._bitable_dict = {}
        if bitable_url:
            self.add_bitable("default", bitable_url)

    @staticmethod
    def _post_req(url, headers, req_body, param=None):
        if param is not None:
            url = url + '?' + urlencode(param)
        print("Post [{}]".format(url))
        data = bytes(json.dumps(req_body), encoding='utf8')
        req = request.Request(url=url, data=data, headers=headers, method='POST')
        try:
            response = request.urlopen(req)
        except Exception as e:
            print("Error", e.read().decode())
            return {}

        rsp_body = response.read().decode('utf-8')
        rsp_dict = json.loads(rsp_body)
        return rsp_dict

    @staticmethod
    def _get_req(url, headers, param=None, method='GET'):
        if param is not None:
            url = url + '?' + urlencode(param)
        print("Get [{}]".format(url))
        req = request.Request(url=url, headers=headers, method=method)
        try:
            response = request.urlopen(req)
        except Exception as e:
            print("Error", e.read().decode())
            return {}

        rsp_body = response.read().decode('utf-8')
        rsp_dict = json.loads(rsp_body)
        return rsp_dict

    def _get_tenant_access_token(self):
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"
        headers = {
            "Content-Type": "application/json"
        }
        req_body = {
            "app_id": self.APP_ID,
            "app_secret": self.APP_SECRET
        }

        rsp_dict = LarkReport._post_req(url, headers, req_body)

        code = rsp_dict.get("code", -1)
        if code != 0:
            print("get tenant_access_token error, code =", code)
            return ""

        return rsp_dict.get("tenant_access_token", "")

    def post_req(self, url: str, headers: dict = None, req_body: dict = None, param=None):
        if headers is None:
            headers = {}
        if req_body is None:
            req_body = {}
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json; charset=utf-8"
        if "Authorization" not in headers:
            headers["Authorization"] = "Bearer " + self.tenant_access_token()
        return LarkReport._post_req(url, headers, req_body, param)

    def delete_req(self, url: str, headers: dict = None, param=None):
        if headers is None:
            headers = {}
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json; charset=utf-8"
        if "Authorization" not in headers:
            headers["Authorization"] = "Bearer " + self.tenant_access_token()
        return LarkReport._get_req(url, headers, param, method="DELETE")

    def get_req(self, url: str, headers: dict = None, param=None):
        if headers is None:
            headers = {}
        if "Authorization" not in headers:
            headers["Authorization"] = "Bearer " + self.tenant_access_token()
        return LarkReport._get_req(url, headers, param)

    def tenant_access_token(self, update=False):
        now_time = datetime.now()
        if update:
            self.storage["data"] = self._get_tenant_access_token()
            self.storage["time"] = now_time
        elif "data" not in self.storage or "time" not in self.storage:
            self.storage["data"] = self._get_tenant_access_token()
            self.storage["time"] = now_time
        elif (now_time - self.storage["time"]).seconds >= 1800:
            self.storage["data"] = self._get_tenant_access_token()
            self.storage["time"] = now_time
        return self.storage["data"]


    # 操作飞书多维表格
    def bitable_create(self, app_token: str, table_id: str, records: list):
        """用于在多维表格上批量新建数据，该方法的参数和 records 的组织方式请参考飞书的 API 文档"""

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_create"
        # API: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-record/batch_create

        rsp_dict = self.post_req(
            url, req_body={
                "records": records
            }
        )

        if rsp_dict.get("code", -1) != 0:
            print("Error[bitable_batch_create]", rsp_dict.get("code"), rsp_dict.get("msg"))
            return []
        return rsp_dict.get("data").get("records")

    def bitable_create_all(self, app_token: str, table_id: str, records: list):
        """用于在多维表格上新建一大批的数据"""
        for k in range(0, len(records), MAX_OPS):
            self.bitable_create(app_token, table_id, records=records[k:k + MAX_OPS])

    def bitable_update(self, app_token: str, table_id: str, records: list):
        """用于在多维表格上批量更新数据，该方法的参数和 records 的组织方式请参考飞书的 API 文档"""

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_update"
        # API: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-record/batch_update

        rsp_dict = self.post_req(
            url, req_body={
                "records": records
            }
        )

        if rsp_dict.get("code", -1) == 0:
            return rsp_dict.get("data").get("records")
        else:
            print("Error[bitable_batch_update]", rsp_dict["msg"])
            return []

    def bitable_update_all(self, app_token: str, table_id: str, records: list):
        r"""
        用于在多维表格上更新一大批的数据
        records: {"records_id": records_id, "fields": {} }
        """
        for k in range(0, len(records), MAX_OPS):
            self.bitable_update(app_token, table_id, records=records[k:k + MAX_OPS])

    def bitable_list(self, app_token: str, table_id: str, filter_dict: dict = None, page_token=""):
        """用于在多维表格上获取记录，该方法的参数请参考飞书的 API 文档"""
        if filter_dict is None:
            filter_dict = {}

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"
        # API: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-record/list
        param = {
            "page_size": MAX_OPS,
            **filter_dict
        }
        if page_token != "":
            param["page_token"] = page_token
        rsp_dict = self.get_req(url, param=param)

        if rsp_dict.get("code", -1) == 0:
            data = rsp_dict.get("data")
            page_token = data.get("page_token", "") if data.get("has_more", False) else False
            return data.get("items"), page_token
        else:
            print("Error[get_records]", rsp_dict["msg"])
            return [], False

    def bitable_list_all(self, app_token: str, table_id: str, filter_dict: dict = None):
        """用于在多维表格上获取所有的记录"""
        page_token = ""
        ret_records = []
        while 1:
            records, page_token = self.bitable_list(app_token, table_id, filter_dict, page_token)
            if records is not None:
                ret_records.extend(records)
            if not page_token:
                break
        return ret_records

    def bitable_delete(self, app_token: str, table_id: str, records: list) -> int:
        """用于在多维表格上批量删除，该方法的参数请参考飞书的 API 文档"""

        if records is None:
            return -1
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_delete"
        # API: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-record/batch_delete

        rsp_dict = self.post_req(
            url, req_body={
                "records": [each["id"] for each in records]
            }
        )

        if rsp_dict.get("code", -1) == 0:
            _ = rsp_dict.get("data")
            return 0
        else:
            print("Error[del_items]", rsp_dict["msg"])
            return rsp_dict.get("code", -1)

    def bitable_delete_all(self, app_token: str, table_id: str, filter_dict: dict = None) -> None:
        """删除某个多维表格中的所有符合条件的数据

        TODO: 可能需要先 list 再 delete 来提升性能

        """

        if filter_dict is None:
            filter_dict = {}
        records, has_more = self.bitable_list(app_token, table_id, filter_dict)
        time.sleep(WAITING_TIME)
        self.bitable_delete(app_token, table_id, records)
        time.sleep(WAITING_TIME)
        if has_more:
            self.bitable_delete_all(app_token, table_id, filter_dict)

    def bitable_field_delete_all(self, app_token: str, table_id: str) -> None:
        """用于在多维表格上删除所有字段，该方法的参数请参考飞书的 API 文档"""

        items = self.bitable_field_list_all(app_token, table_id)
        for field in items:
            self.bitable_field_delete(app_token, table_id, field["field_id"])

    def bitable_field_delete(self, app_token: str, table_id: str, field_id: str) -> None:
        """用于在多维表格上删除字段，该方法的参数请参考飞书的 API 文档"""

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/fields/{field_id}"
        # API: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-field/delete

        rsp_dict = self.delete_req(url)
        if rsp_dict.get("code", -1) != 0:
            print("Error[bitable_delete_field]", rsp_dict.get("code"), rsp_dict.get("msg"))

    def bitable_field_list_all(self, app_token: str, table_id: str) -> list:
        """用于在多维表格上列出所有字段，该方法的参数请参考飞书的 API 文档"""

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/fields"
        # API: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-field/list
        rsp_dict = self.get_req(url)
        if rsp_dict.get("code", -1) != 0:
            print("Error[bitable_list_all_fields]", rsp_dict.get("code"), rsp_dict.get("msg"))
        return rsp_dict.get("data", {"items": []}).get("items")

    def bitable_field_create(
        self, app_token: str, table_id: str, field_name: str, field_type: int = 1,
        field_property: dict = None
    ) -> None:
        """用于在多维表格上增加字段，该方法的参数请参考飞书的 API 文档"""
        if field_name == "时间":
            field_type = 1001
            field_property = {
                "date_formatter": "yyyy-MM-dd HH:mm"
            }
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/fields"
        # API: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-field/create
        if field_property is not None:
            if "formatter" in field_property:
                formatter = str(field_property["formatter"])
                del field_property["formatter"]
                rsp_dict = self.post_req(
                    url, req_body={
                        "field_name": field_name,
                        "type": field_type,
                        "formatter": formatter,
                        "property": field_property
                    }
                )
            else:
                rsp_dict = self.post_req(
                    url, req_body={
                        "field_name": field_name,
                        "type": field_type,
                        "property": field_property
                    }
                )
        else:
            rsp_dict = self.post_req(
                url, req_body={
                    "field_name": field_name,
                    "type": field_type,
                }
            )
        if rsp_dict.get("code", -1) != 0:
            if rsp_dict.get("code") != 1254014:
                print("Error[bitable_create_field]", rsp_dict.get("code"), rsp_dict.get("msg"))

    def add_bitable(self, table_name: str, link: str, comment: str = "") -> None:
        """用于存储多维表格的配置信息

        :param table_name: 开发者定义的多维表格名
        :param link: 浏览器中多维表格的地址
        :param comment: 额外的备注信息
        :return:
        """
        if table_name in self._bitable_dict:
            print("Error! Table name {} has been saved in config.".format(table_name))
            return
        link_end = link.split("/")[-1]
        app_token = link_end.split("?")[0]
        params = link_end.split("?")[-1].split('&')
        table_id = ""
        for param in params:
            try:
                if param.split("=")[0] == 'table':
                    table_id = param.split("=")[1]
            except IndexError:
                pass
        if table_id == "":
            print("Error! Table id is not been found")
            return
        self._bitable_dict[table_name] = {
            "app_token": app_token,
            "table_id": table_id,
            "comment": comment
        }

    def bitable(self, table_name: str="default"):
        """

        :param table_name: 开发者定义的多维表格名
        :return: app_token, table_id
        """
        if table_name not in self._bitable_dict:
            raise KeyError("未找到名为{}的多维表格".format(table_name))
        item = self._bitable_dict[table_name]
        return item["app_token"], item["table_id"]

    def send(self, res: dict| list, table_name: str="default"):
        if isinstance(res, list):
            for res_item in res:
                self.send(res_item, table_name)
                time.sleep(WAITING_TIME)
        else:
            print("*" * 80)
            print("正在使用飞书 API 发送数据, 下面会进行若干 Post 请求, 请保持网络通常")
            if "time" not in res and "时间" not in res:
                res = {
                    **res,
                    "时间": round(datetime.now().timestamp() * 1000)
                }
            app_token, table_id = self.bitable(table_name)
            fields = set([f["field_name"] for f in self.bitable_field_list_all(app_token, table_id)])
            for key, value in res.items():
                if inspect.isfunction(value):
                    res[key] = str(value.__name__)
                elif isinstance(value, int) or isinstance(value, float) or isinstance(value, bool):
                    res[key] = value
                else:
                    res[key] = str(value)
                if key not in fields:
                    if isinstance(value, int) or isinstance(value, float):
                        field_type = 2
                    else:
                        field_type = 1
                    self.bitable_field_create(app_token, table_id, key, field_type)
            print(res)
            self.bitable_create_all(app_token, table_id, [{"fields": res}])
            print("*" * 80)
            print()