from django.contrib import admin
from .models import Document

class DocumentAdmin(admin.ModelAdmin):
    search_fields = ['subject']

admin.site.register(Document, DocumentAdmin)
