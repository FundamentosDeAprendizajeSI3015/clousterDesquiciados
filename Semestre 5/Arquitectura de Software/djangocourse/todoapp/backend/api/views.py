from rest_framework import generics,permissions
from .serializers import TodoSerializer,TodoToggleCompleteSerializer
from todo.models import ToDo
from django.db import IntegrityError
from django.contrib.auth.models import User
from rest_framework.parsers import JSONParser
from rest_framework.authtoken.models import Token
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate
from rest_framework.decorators import api_view, permission_classes, authentication_classes, parser_classes
from rest_framework.permissions import AllowAny
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework import status


class TodoList(generics.ListAPIView):
    serializer_class = TodoSerializer

    def get_queryset(self):   
        user = self.request.user
        return ToDo.objects.filter(user=user).order_by('-created')

class TodoListCreate(generics.ListCreateAPIView):
    # ListAPIView requires two mandatory attributes, serializer_class and
    # queryset.
    # We specify TodoSerializer which we have earlier implemented
    serializer_class = TodoSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        return ToDo.objects.filter(user=user).order_by('-created')

    def perform_create(self, serializer):
        # serializer holds a django model
        serializer.save(user=self.request.user)
        
class TodoRetrieveUpdateDestroy(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = TodoSerializer
    permission_classes = [permissions.IsAuthenticated]
    def get_queryset(self):
        user = self.request.user
        # user can only update, delete own posts
        return ToDo.objects.filter(user=user) 
    
class TodoToggleComplete(generics.UpdateAPIView):
    serializer_class = TodoToggleCompleteSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        return ToDo.objects.filter(user=user)

    def perform_update(self, serializer):
        serializer.instance.completed = not (serializer.instance.completed)
        serializer.save()
        
@api_view(['POST'])
@authentication_classes([])                 # No exige auth para crear/entrar
@permission_classes([AllowAny])             # Público
@parser_classes([JSONParser, FormParser, MultiPartParser])  # Acepta JSON y form
def signup(request):
    username = request.data.get('username')
    password = request.data.get('password')

    if not username or not password:
        return Response({'error': 'username and password are required'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        user = User.objects.create_user(username=username, password=password)
    except IntegrityError:
        return Response({'error': 'username taken. choose another username'}, status=status.HTTP_400_BAD_REQUEST)

    token, _ = Token.objects.get_or_create(user=user)
    return Response({'token': str(token)}, status=status.HTTP_201_CREATED)


@api_view(['POST'])
@authentication_classes([])                 
@permission_classes([AllowAny])
@parser_classes([JSONParser, FormParser, MultiPartParser])
def login(request):
    username = request.data.get('username')
    password = request.data.get('password')

    if not username or not password:
        return Response({'error': 'username and password are required'}, status=status.HTTP_400_BAD_REQUEST)

    user = authenticate(request, username=username, password=password)
    if user is None:
        return Response({'error': 'unable to login. check username and password'}, status=status.HTTP_400_BAD_REQUEST)

    token, _ = Token.objects.get_or_create(user=user)
    return Response({'token': str(token)}, status=status.HTTP_201_CREATED)