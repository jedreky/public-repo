# external key pair must be specified
EXTERNAL_SSH_KEY = None
EXTERNAL_SSH_KEY = "external_ssh_key"
assert EXTERNAL_SSH_KEY is not None, "Key pair for the client must be specified"

# internal key pair is optional
INTERNAL_SSH_KEY = None
INTERNAL_SSH_KEY = "internal_ssh_key"

# fill in once AMIs have been built
WHISPER_AMIS = {
    ("x86", "cpu"): "ami-013254df26fce0a79",
    ("x86", "gpu"): "ami-05ed6bb726279cf59",
}
