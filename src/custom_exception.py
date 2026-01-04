import traceback
import sys  # untuk menunjukkan error pesan

class CustomException(Exception): #inheritance (pewarisan): menambahkan custom error kemudian menambahkannya kedalam error utama di python (exception) 

    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = self.get_detail_error_message(error_message, error_detail)

    @staticmethod
    def get_detail_error_message(error_message, error_detail:sys):

        _, _, exc_tb = traceback.sys.exc_info() # import traceback diatas untuk menampilkan errornya, kami ingin mengambil pesan yg terakhir saja maka dari itu digunakan _, _, exc_tb -> yang ketigas disimpan didalah exc_tb
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return f'Error in file {file_name}, line {line_number} : {error_message}'
    
    def __str__(self):
        return self.error_message
