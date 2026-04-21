from rest_framework import serializers
from todo.models import ToDo

class TodoSerializer(serializers.ModelSerializer):
    created = serializers.ReadOnlyField()
    completed = serializers.ReadOnlyField()

    class Meta:
        model = ToDo
        fields = ['id', 'title', 'memo', 'created', 'completed']

class TodoToggleCompleteSerializer(serializers.ModelSerializer):
    class Meta:
        model = ToDo
        fields = ['id', 'completed']