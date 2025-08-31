import os
import sys
import glob
import win32com.client as win32

XL_OPENXML_WORKBOOK = 51  # .xlsx
XL_REPAIR_FILE = 1  # 尝试修复打开


def convert_dir(root, recurse=True, delete_original=True):
    excel = win32.DispatchEx("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False

    pattern = "**/*.xls" if recurse else "*.xls"
    files = glob.glob(os.path.join(root, pattern), recursive=recurse)

    ok = fail = 0
    for src in files:
        if not src.lower().endswith(".xls"):
            continue
        dst = os.path.splitext(src)[0] + ".xlsx"
        if os.path.exists(dst):
            print(f"[SKIP] {dst} 已存在")
            continue
        wb = None
        try:
            wb = excel.Workbooks.Open(src, CorruptLoad=XL_REPAIR_FILE)
            wb.CheckCompatibility = False
            wb.SaveAs(dst, FileFormat=XL_OPENXML_WORKBOOK)
            wb.Close(SaveChanges=True)
            print(f"[OK] {src} -> {dst}")
            ok += 1

            # 删除原始 .xls
            if delete_original:
                os.remove(src)
                print(f"[DEL] 删除原文件：{src}")

        except Exception as e:
            print(f"[FAIL] {src}: {e}")
            if wb is not None:
                wb.Close(SaveChanges=False)
            fail += 1

    excel.Quit()
    print(f"完成。成功 {ok} 个，失败 {fail} 个。")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    convert_dir(root, recurse=True, delete_original=True)
