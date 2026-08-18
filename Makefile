.PHONY: bootstrap build start stop test verify doctor demo pair set benchmark frontend-check frontend-lock cli api logs clean release

bootstrap:
	./scripts/bootstrap.sh

build:
	docker compose build

start:
	./scripts/start.sh

stop:
	./scripts/stop.sh

test:
	./scripts/test.sh

verify:
	./scripts/verify.sh

doctor:
	./scripts/doctor.sh

demo:
	./scripts/demo.sh

benchmark:
	./scripts/benchmark.sh audio --compare-serial

frontend-check:
	./scripts/check-frontend.sh

frontend-lock:
	./scripts/lock-frontend.sh

cli:
	./scripts/djai.sh --help

api:
	docker compose up api

logs:
	docker compose logs -f api frontend

clean:
	./scripts/clean.sh

release:
	./scripts/package-release.sh
