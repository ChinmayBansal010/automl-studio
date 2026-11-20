import asyncio
import tempfile
import time
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import HTTPException
import httpx
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import main


class AuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.private_key = Ed25519PrivateKey.generate()
        cls.public_key = cls.private_key.public_key()

    def make_token(self, **overrides):
        now = int(time.time())
        claims = {
            "sub": "user-123",
            "iat": now,
            "exp": now + 60,
            "iss": "https://auth.example",
            "aud": "https://auth.example",
            "emailVerified": True,
        }
        claims.update(overrides)
        return jwt.encode(claims, self.private_key, algorithm="EdDSA", headers={"kid": "test-key"})

    def test_api_routes_require_a_bearer_token(self):
        async def request():
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/api/workspace")

        with patch.object(main, "JWKS_CLIENT", SimpleNamespace()):
            response = asyncio.run(request())
        self.assertEqual(response.status_code, 401)

    def test_health_is_public(self):
        async def request():
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/api/health")

        response = asyncio.run(request())
        self.assertEqual(response.status_code, 200)

    def test_valid_jwt_returns_authenticated_user(self):
        jwks = SimpleNamespace(get_signing_key_from_jwt=lambda token: SimpleNamespace(key=self.public_key))
        with patch.object(main, "JWKS_CLIENT", jwks), patch.object(main, "AUTH_ORIGIN", "https://auth.example"):
            self.assertEqual(main.current_user(f"Bearer {self.make_token()}"), "user-123")

    def test_expired_or_wrong_issuer_jwt_is_rejected(self):
        jwks = SimpleNamespace(get_signing_key_from_jwt=lambda token: SimpleNamespace(key=self.public_key))
        with patch.object(main, "JWKS_CLIENT", jwks), patch.object(main, "AUTH_ORIGIN", "https://auth.example"):
            with self.assertRaises(HTTPException) as expired:
                main.current_user(f"Bearer {self.make_token(exp=int(time.time()) - 1)}")
            self.assertEqual(expired.exception.status_code, 401)

            with self.assertRaises(HTTPException) as wrong_issuer:
                main.current_user(f"Bearer {self.make_token(iss='https://attacker.example')}")
            self.assertEqual(wrong_issuer.exception.status_code, 401)

    def test_unverified_email_jwt_is_rejected(self):
        jwks = SimpleNamespace(get_signing_key_from_jwt=lambda token: SimpleNamespace(key=self.public_key))
        token = self.make_token(emailVerified=False)
        with patch.object(main, "JWKS_CLIENT", jwks), patch.object(main, "AUTH_ORIGIN", "https://auth.example"):
            with self.assertRaises(HTTPException) as denied:
                main.current_user(f"Bearer {token}")
            self.assertEqual(denied.exception.status_code, 403)

    def test_auth_proxy_rewrites_upstream_cookie_for_frontend_origin(self):
        upstream_request = {}

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def request(self, method, url, **kwargs):
                upstream_request.update(method=method, url=url, headers=kwargs["headers"])
                return httpx.Response(
                    200,
                    headers=[
                        ("content-type", "application/json"),
                        ("set-cookie", "session=token; Domain=auth.example; Path=/db/auth; HttpOnly; Secure"),
                    ],
                    content=b"{}",
                )

        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        request = Request({
            "type": "http",
            "method": "POST",
            "path": "/auth/sign-in/email",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"cookie", b"session=old"),
                (b"origin", b"https://studio.example"),
            ],
        }, receive)
        with patch.object(main, "NEON_AUTH_BASE_URL", "https://auth.example/db/auth"), patch.object(main.httpx, "AsyncClient", return_value=FakeClient()):
            response = asyncio.run(main.auth_proxy("sign-in/email", request))

        self.assertEqual(upstream_request["url"], "https://auth.example/db/auth/sign-in/email")
        self.assertEqual(upstream_request["headers"]["origin"], "https://studio.example")
        self.assertIn("Path=/auth", response.headers["set-cookie"])
        self.assertNotIn("Domain=", response.headers["set-cookie"])

    def test_user_workspaces_cannot_read_each_others_datasets(self):
        dataset_id = str(uuid4())
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(main, "ROOT", Path(temp_dir)):
                data_dir, _, _, _ = main.user_workspace("owner")
                (data_dir / f"{dataset_id}.csv").write_text("value\nsecret\n", encoding="utf-8")
                main.datasets.pop(("owner", dataset_id), None)

                self.assertEqual(main.get_df(dataset_id, "owner").iloc[0, 0], "secret")
                with self.assertRaises(HTTPException) as denied:
                    main.get_df(dataset_id, "other")
                self.assertEqual(denied.exception.status_code, 404)
                main.datasets.pop(("owner", dataset_id), None)


if __name__ == "__main__":
    unittest.main()
