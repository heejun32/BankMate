# Generated by Django 3.1.3 on 2021-11-13 02:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bankmate', '0002_answer_author'),
    ]

    operations = [
        migrations.AddField(
            model_name='answer',
            name='modify_date',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='document',
            name='modify_date',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
