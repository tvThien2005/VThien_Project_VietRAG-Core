import tempfile
from pathlib import Path
from unittest import TestCase

from fastapi.testclient import TestClient

from api.auth import UserStore, decode_access_token, get_user_store, issue_access_token
from api.main import app


class TestAuth(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = UserStore(Path(self.temp_dir.name) / "users.sqlite3")

    def tearDown(self):
        app.dependency_overrides.clear()
        self.temp_dir.cleanup()

    def test_passwords_are_hashed_and_token_identifies_registered_user(self):
        user = self.store.register(
            email="thien@example.com",
            password="123456",
            full_name="Văn Thiện",
        )

        self.assertEqual(self.store.authenticate(email=user.email, password="123456"), user)
        self.assertIsNone(self.store.authenticate(email=user.email, password="wrong-password"))
        claims = decode_access_token(issue_access_token(user))
        self.assertEqual(claims["sub"], user.user_id)

    def test_auth_routes_issue_token_and_protect_documents(self):
        app.dependency_overrides[get_user_store] = lambda: self.store
        client = TestClient(app)

        registered = client.post(
            "/auth/register",
            json={"email": "user@example.com", "password": "123456", "full_name": "User One"},
        )
        self.assertEqual(registered.status_code, 201)

        logged_in = client.post(
            "/auth/login",
            json={"email": "user@example.com", "password": "123456"},
        )
        self.assertEqual(logged_in.status_code, 200)
        token = logged_in.json()["access_token"]

        self.assertEqual(client.get("/documents").status_code, 401)
        self.assertEqual(
            client.get("/documents", headers={"Authorization": f"Bearer {token}"}).status_code,
            200,
        )
