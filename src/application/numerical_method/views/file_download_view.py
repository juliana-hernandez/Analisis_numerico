import os
from django.http import FileResponse, Http404
from django.views import View
from django.conf import settings


class FileDownloadView(View):

    def get(self, request, *args, **kwargs):
        file_path = os.path.join(
            settings.BASE_DIR, "static/img/numerical_method/function_plot.svg"
        )

        response = FileResponse(open(file_path, "rb"), content_type="image/svg+xml")
        response["Content-Disposition"] = 'attachment; filename="function_plot.svg"'
        return response
